# Design a bot frame work
import os
import time
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from colorama import Fore, Style
load_dotenv()


class BOT_FRAMEWORK():
    def __init__(self, configuration_filepath: str, vector_database) -> None:

        self.LOGGER = self.get_logger()
        self.config = self.load_config(configuration_filepath)
        self.llm = self.get_llm(model_name=self.config['chat_model_name'])
        self.system_prompt = self.get_system_behavior_prompt()
        self.message_manager = {
            "history": [self.system_prompt],
            "questions": [],
            "answers": []
        }
        # try to give odd numbers
        self.max_history_limit = self.config['max_history_limit']
        self.vector_database = vector_database

    def get_logger(self, filename='logs/chat_log'):
        from logging import getLogger, DEBUG, FileHandler, Formatter

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        logger = getLogger(__name__)
        logger.setLevel(DEBUG)
        handler = FileHandler(filename=f"{filename}.log")
        handler.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.info(f"Time: {time.asctime(time.localtime())} \n")
        return logger

    def load_config(self, configuration_filepath):
        with open(configuration_filepath, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get_llm(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1, max_tokens: int = 1024):
        """Initiate llm with desired parameters

        Args:
            model_name (str, optional): LLM model name. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): randomness in LLM generation. Defaults to 0..
            max_tokens (int, optional): Max tokens that can generate by the llm model. Defaults to 1024.

        Returns:
            LLM for response generation
        """
        llm = ChatOpenAI(
            model=model_name,  # gpt-4o-mini
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens
        )
        # llm = AzureChatOpenAI(
        #     azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        #     openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        #     temperature=temperature,
        #     max_tokens=max_tokens
        # )

        return llm

    def get_system_behavior_prompt(self) -> SystemMessage:
        """Prompt defined for the system behavior.
        """
        system_prompt = self.config['system_prompt']
        system_prompt = SystemMessage(content=system_prompt)
        return system_prompt

    def get_context_with_similarity(self, query: str, k: int = 3) -> tuple:
        """Search from vector database and get relevant contexts with similarity scores for the query.
        Args:
            query (str): The user query.
            k (int): Number of top contexts to retrieve.
        Returns:
            tuple: (contexts_list, similarity_scores_list) - Two lists in matching order
        """
        # Extract chunks with similarity scores
        results = self.vector_database.extract_chunks(
            query, k=k, similarity_score=True)
        print(f"Results with scores: {results}")

        # Separate contexts and scores into two lists while deduplicating
        seen = set()
        contexts_list = []
        similarity_scores = []

        for doc, score in results:
            page_content = doc.page_content
            if page_content not in seen:
                contexts_list.append(page_content)
                similarity_scores.append(score)
                seen.add(page_content)

        self.LOGGER.info(f"CONTEXTS: {contexts_list}")
        self.LOGGER.info(f"SIMILARITY_SCORES: {similarity_scores}")

        return contexts_list, similarity_scores

    def get_context(self, query: str) -> str:
        """Search from vector database and get relevent contexts for the query.
        Args:
            query (str):
        Returns:
            context (str): context related to query
        """
        contexts = self.vector_database.extract_chunks(
            query, k=3, similarity_score=False)
        print(f"Contexts: {contexts}")
        contexts = [txt.page_content for txt in contexts]
        contexts = set(contexts)
        contexts = "\n".join([txt for txt in contexts])

        self.LOGGER.info(f"CONTEXT: {contexts}")

        return contexts

    def add_context_related_to_query(self, query: str) -> str:
        """Add relevent context related to the query using retrieval method.

        Args:
            query (str): user query
        Returns:   
            Modified query with context.
        """

        CONTEXT = self.get_context(query)
        query = f"Using the CONTEXT below, answer the QUESTION \n \
        CONTEXT: {CONTEXT} \n QUESTION: {query} \n"
        return query

    def format_query(self, query: str) -> list:
        """Formate the query which can be pass to llm, also track previous question answer on the loop for relevence.
        * Augment the query with the prompt for adding context related to query and. [Invoke augment_prompt function]
        * Give list of message for continuous context. simple length check in message manager and take the latest message to pass to llm.

        Args:
            query (str): user query

        Returns:
            Formated query to pass it to the llm
        """

        # adding the raw questions in the message manager
        self.message_manager['questions'].append(query)
        query = self.add_context_related_to_query(query=query)
        query = HumanMessage(content=query)
        self.message_manager['history'].append(query)

        # taking previous message history for
        if len(self.message_manager["history"]) > self.max_history_limit:
            messages = self.message_manager["history"][-self.max_history_limit:].copy()
            messages = [self.system_prompt] + messages
        else:
            messages = self.message_manager["history"]

        # print(f"All questions: {self.get_all_questions()}")
        return messages

    def get_llm_response(self, query: str) -> str:
        """Generated response for the given single query.
        * Also add the answer into message manager.

        Args:
            query (str)
        Returns:   
            Response from the llm
        """
        # self.LOGGER.info(f"\033[1;34")
        self.LOGGER.info(f"\n {Fore.BLUE} New Query: {Style.RESET_ALL}")
        self.LOGGER.info(f"User_Query: {query}")

        messages = self.format_query(query=query)
        llm_response = self.llm(messages=messages)
        # llm_response = self.llm(input=messages)
        # .content
        # print(f"\n {Fore.GREEN} LLM Response: {Style.RESET_ALL}")
        self.LOGGER.info(f"Response: {llm_response}")
        llm_response = llm_response.content
        ########
        self.message_manager['answers'].append(llm_response)
        # adding the llm response in message manger for continuous messaging.
        self.message_manager['history'].append(AIMessage(content=llm_response))
        return llm_response

    def get_all_questions(self):
        return self.message_manager['questions']


# if __name__ == "__main__":
#     bot = BOT_FRAMEWORK()
#     bot.vector_database = bot.get_vector_database(create_new=True)
#     bot.add_data_into_vector_db(filepath="knowledge_base.txt", create_new=True)
