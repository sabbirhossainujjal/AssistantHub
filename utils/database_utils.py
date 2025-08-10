from datetime import datetime
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from qdrant_client import QdrantClient, models
import os
import yaml
from dotenv import load_dotenv
load_dotenv()

embedding_size_mapping = {
    # 'gpt-4o-mini': 1536,
    # 'gpt-4o': 1536,
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    "models/embedding-001": 768,  # gemeni model

}


class VectorDatabase:
    # , embedding_type:str, embedding_size:int, model_name:str
    def __init__(self, configuration_filepath: str):
        self.config = self.load_config(
            configuration_filepath=configuration_filepath
        )
        self.qdrant_client = QdrantClient(
            url=self.config['database_url'],
            # use environment variable for API key
            api_key=os.getenv("QDRANT_API_KEY")
            # if database is key protected use api key
        )

        self.collection_name = self.config['collection_name']
        self.embedding_type = self.config['embedding_type']
        self.embedding_model_name = self.config['embedding_model_name']
        self.embedding_size = embedding_size_mapping[self.embedding_model_name]

        self.embedding_model = self.get_embedding_model(
            self.embedding_type,
            self.embedding_model_name
        )

        if self.qdrant_client.collection_exists(collection_name=self.collection_name):
            self.database = self.load_database(self.collection_name)
        else:
            self.database = None

    def load_config(self, configuration_filepath="config.yml"):
        with open(configuration_filepath, 'r') as file:
            config = yaml.safe_load(file)

        return config

    # , embedding_type:str='openai', model_name:str=None
    def create_vector_database(self):
        vector_config = models.VectorParams(
            size=self.embedding_size,
            distance=models.Distance.COSINE
        )
        if not self.qdrant_client.collection_exists(collection_name=self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_config,
            )
            print(
                f"Vector Database created with collection name {self.collection_name}")
            self.database = self.load_database(
                collection_name=self.collection_name
            )

        else:
            self.database = self.load_database(
                collection_name=self.collection_name
            )
            print(
                f"Vector Database collection with name {self.collection_name} is already exists. Loaded exiting database. To create new database, please delete the database or create database with new name")

        return self.database

    def check_collection_exists(self, collection_name: str):
        """Check if the collection exists in the Qdrant database

        Args:
            collection_name (str): name of the collection to check

        Returns:
            bool: True if the collection exists, False otherwise
        """
        return self.qdrant_client.collection_exists(collection_name=collection_name)

    def get_embedding_model(self, embedding_type: str, model_name: str = None):
        """get embedding model for creating 

        Args:
            embedding_type (str): type of the embedding model. options are- 'openai', 'ollama', 'huggingface'
            model_name (str): Actual name of the embedding model.options are - 'gpt-4o', 'sentence-transformers/all-MiniLM-L6-v2', 'llama2' and all other ollama models and huggingface models.

        Returns:
            Desired embedding or raises value error if embedding not found.
        """
        if embedding_type == 'openai':
            return OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_AI_KEY"))

        elif embedding_type == 'azure_openai':
            return AzureOpenAIEmbeddings(
                # os.getenv("DEPLOYMENT_NAME"),
                model=self.embedding_model_name,
                # azure_deployment=os.getenv("DEPLOYMENT_NAME"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION")
            )
        # elif embedding_type == 'ollama':
        #     return OllamaEmbeddings(base_url=os.getenv("OLLAMA_BASE_URL"), model=model_name)

        elif embedding_type == 'gemini':
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                api_key=os.getenv("GOOGLE_API_KEY")
            )
        # elif embedding_type == 'huggingface':
        #     return HuggingFaceEmbeddings(model_name=model_name)
        else:
            raise ValueError("Unsupported embedding type")

    def load_database(self, collection_name: str = None):
        """Load an existing vector database

        Args:
            embedding_type (str, optional): type of the embedding model. options are- 'openai', 'ollama', 'huggingface'. Defaults to 'openai'.
            model_name (_type_, optional): Actual name of the embedding model. Defaults to None.

        Returns:
            Qdrant Vector Database
        """
        if collection_name == None:
            collection_name = self.collection_name

        if self.qdrant_client.collection_exists(collection_name=collection_name):
            # load the database
            self.database = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                embedding=self.embedding_model,
            )
            print(
                f"Vector database with collection name {collection_name} is loaded")
            return self.database
        else:
            raise ValueError(
                f"Collection `{collection_name}` doesn't exist! Please create the collection first.")

    def delete_database(self, collection_name: str = None):
        """Delete the vector database

        Args:
            collection_name (str, optional): name of the collection. Defaults to None.

        Returns:
            None
        """
        if collection_name is None:
            collection_name = self.collection_name

        if self.qdrant_client.collection_exists(collection_name=collection_name):
            self.qdrant_client.delete_collection(
                collection_name=collection_name)
            print(f"Collection {collection_name} deleted successfully.")
        else:
            raise ValueError(
                f"Collection {collection_name} does not exist. Please create the collection first.")

    def add_data(self, chunks: List[str], metadatas: List[dict] = None):
        """Add data into vector database

        Args:
            chunks (List[str]): list of chunks

        Returns:
            The same Qdrant database with added data
        """
        if not self.qdrant_client.collection_exists(collection_name=self.collection_name):
            raise ValueError(
                f"Collection {self.collection_name} does not exist. Please create the collection first.")

        current_time = datetime.utcnow().isoformat()
        if metadatas is None:
            metadatas = [{} for _ in chunks]

        elif len(metadatas) != len(chunks):
            raise ValueError(
                "The number of metadata items must match the number of chunks")

        for meta in metadatas:
            meta['timestamp'] = current_time

        self.database.add_texts(texts=chunks, metadatas=metadatas)
        print(f"Added {len(chunks)} chunks to the vector database.")

        return self.database

    def extract_chunks(self, query: str, k: int = 3, similarity_score: bool = True) -> List[str]:

        if similarity_score:
            chunks = self.database.similarity_search_with_score(
                query=query, k=k)
        else:
            chunks = self.database.similarity_search(query=query, k=k)
        return chunks


# if __name__ == "__main__":
#     print("Running demo")
#     vector_database = VectorDatabase(configuration_filepath="../config.yml")
#     vector_db = vector_database.create_vector_database()
#     vector_db = vector_database.add_data(["hello data"])
