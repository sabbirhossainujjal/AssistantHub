import os
import docx
import PyPDF2
import nltk
import tiktoken
import json
from datetime import datetime
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Dict, Any, Callable


def read_file(filepath: str = "./data/demo.txt"):
    """Read file content based on file extension.

    Args:
        filepath (str): Path to the file

    Returns:
        str: File content as text
    """
    _, file_extension = os.path.splitext(filepath)
    file_extension = file_extension.lower()

    # Read the file based on its extension
    if file_extension == '.txt':
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()

    elif file_extension == '.pdf':
        text = ""
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    elif file_extension in ['.doc', '.docx']:
        doc = docx.Document(filepath)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def load_files_from_directory(directory_path: str, file_extensions: List[str] = ['.txt', '.pdf', '.docx']) -> Dict[str, str]:
    """Load all supported files from the specified directory.

    Args:
        directory_path: Path to the directory containing files
        file_extensions: List of file extensions to include

    Returns:
        Dictionary mapping file names to their content
    """
    files_content = {}
    directory = Path(directory_path)

    if not directory.exists() or not directory.is_dir():
        raise ValueError(
            f"The specified path '{directory_path}' does not exist or is not a directory")

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in file_extensions:
            try:
                content = read_file(str(file_path))
                files_content[file_path.name] = content
                print(f"Loaded file: {file_path.name}")
            except Exception as e:
                print(f"Error loading file {file_path.name}: {str(e)}")

    if not files_content:
        print(
            f"No files with extensions {file_extensions} found in {directory_path}")
    else:
        print(f"Loaded {len(files_content)} files from {directory_path}")

    return files_content


def chunk_text(texts: str, chunk_option: str, chunk_size: int = 1000, chunk_overlap: int = 200, splitter: str = "##Info") -> List[str]:
    """Create chunks from text using specified method.

    Args:
        texts (str): Text to be chunked
        chunk_option (str): chunking options. 'character', 'token', 'sentence', or 'custom'
        chunk_size (int, optional): Size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between consecutive chunks. Defaults to 200.
        splitter (str, optional): Custom splitter for 'custom' option. Defaults to "##Info".

    Returns:
        List[str]: Chunked text
    """
    if chunk_option == 'character':
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return chunker.split_text(texts)

    elif chunk_option == 'token':
        overlap_tokens = max(1, chunk_overlap // 4)
        return split_text_by_tokens(texts, max_tokens_per_chunk=chunk_size, overlap_tokens=overlap_tokens)

    elif chunk_option == 'sentence':
        max_sentences = max(1, chunk_size // 100)
        overlap_sentences = max(0, chunk_overlap // 100)
        return split_text_by_sentences(texts, max_sentences_per_chunk=max_sentences, overlap_sentences=overlap_sentences)

    elif chunk_option == 'custom':
        return [txt.strip() for txt in texts.split(splitter) if txt.strip()]

    else:
        raise ValueError(
            "Invalid chunk option. Choose 'character', 'token', 'sentence', or 'custom'.")


def split_text_by_tokens(text: str, max_tokens_per_chunk: int = 1000, overlap_tokens: int = 100) -> List[str]:
    """Split text into chunks based on token count using tiktoken with overlap.

    Args:
        text: Text to split
        max_tokens_per_chunk: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks

    Returns:
        List of text chunks with overlap
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if overlap_tokens >= max_tokens_per_chunk:
        raise ValueError(
            "Overlap tokens must be less than max_tokens_per_chunk")

    if len(tokens) <= max_tokens_per_chunk:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens_per_chunk, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))

        if end >= len(tokens):
            break
        start += (max_tokens_per_chunk - overlap_tokens)

    return chunks


def split_text_by_sentences(text: str, max_sentences_per_chunk: int = 10, overlap_sentences: int = 2,
                            punctuation_marks: List[str] = ['.', '!', '?', ';']) -> List[str]:
    """Split text into chunks based on sentence boundaries using punctuation marks.

    Args:
        text: Text to split
        max_sentences_per_chunk: Maximum number of sentences per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        punctuation_marks: List of punctuation marks that define sentence boundaries

    Returns:
        List of text chunks split by sentences
    """
    import re

    if overlap_sentences >= max_sentences_per_chunk:
        raise ValueError(
            "Overlap sentences must be less than max_sentences_per_chunk")

    pattern = r'[' + ''.join(re.escape(p) for p in punctuation_marks) + r']\s*'
    sentences = [s.strip() + '.' for s in re.split(pattern, text) if s.strip()]

    if not sentences:
        return [text]

    if len(sentences) <= max_sentences_per_chunk:
        return [' '.join(sentences)]

    chunks = []
    start = 0

    while start < len(sentences):
        end = min(start + max_sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[start:end]
        chunk_text = ' '.join(chunk_sentences)

        if not chunk_text.endswith(tuple(punctuation_marks)):
            chunk_text = chunk_text.rstrip('.') + '.'

        chunks.append(chunk_text)

        if end >= len(sentences):
            break
        start += (max_sentences_per_chunk - overlap_sentences)

    return chunks


def _efficient_chunking(text: str, target_size: int = 1000, method: str = "auto") -> List[str]:
    """Automatically choose the best chunking method based on text characteristics.

    Args:
        text: Text to chunk
        target_size: Target chunk size
        method: 'auto', 'character', 'token', 'sentence', or 'custom'

    Returns:
        List of text chunks
    """
    if method == "auto":
        text_len = len(text)
        sentence_count = text.count('.') + text.count('!') + text.count('?')

        if text_len > 50000:
            method = "token"
        elif sentence_count > 100:
            method = "sentence"
        elif text_len > 10000:
            method = "token"
        else:
            method = "character"

    # Use appropriate overlap based on target size
    chunk_overlap = min(200, target_size // 5)  # Max 20% overlap
    return chunk_text(text, chunk_option=method, chunk_size=target_size, chunk_overlap=chunk_overlap)


def process_directory(
    directory_path: str,
    chunk_method: str = "auto",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    preprocess_fn: Callable[[str], Any] = None,
    output_file_path: str = None,
    file_extensions: List[str] = ['.txt', '.pdf', '.docx']
) -> List[Dict[str, Any]]:
    """Process all supported files in a directory, splitting them into chunks.

    Args:
        directory_path: Path to directory containing files
        chunk_method: Chunking method ('auto', 'character', 'token', 'sentence', 'custom')
        chunk_size: Size of each chunk (tokens for 'token', characters for others)
        chunk_overlap: Overlap between chunks
        preprocess_fn: Optional function to process each text chunk
        output_file_path: Optional path to save the combined output
        file_extensions: List of file extensions to process

    Returns:
        List of dictionaries containing processed results
    """
    print(f"Starting to process files in: {directory_path}")

    files_content = load_files_from_directory(directory_path, file_extensions)
    all_processed_results = []

    for file_name, content in files_content.items():
        print(f"\nProcessing file: {file_name}")

        # Choose chunking method with proper overlap handling
        if chunk_method == "token":
            # Convert to approximate token overlap
            overlap_tokens = max(1, chunk_overlap // 4)
            chunks = split_text_by_tokens(
                content, max_tokens_per_chunk=chunk_size, overlap_tokens=overlap_tokens)
        elif chunk_method == "sentence":
            # Convert to sentence count
            max_sentences = max(1, chunk_size // 100)
            overlap_sentences = max(0, chunk_overlap // 100)
            chunks = split_text_by_sentences(
                content, max_sentences_per_chunk=max_sentences, overlap_sentences=overlap_sentences)
        else:
            chunks = chunk_text(content, chunk_option=chunk_method,
                                chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        print(f"Split into {len(chunks)} chunks using {chunk_method} method")

        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)} of {file_name}")
            try:
                if preprocess_fn is not None:
                    processed_result = preprocess_fn(chunk)
                else:
                    processed_result = chunk

                result_with_metadata = {
                    "file_name": file_name,
                    "chunk_index": i,
                    "chunk_method": chunk_method,
                    "chunk_size": len(chunk),
                    "chunk_overlap": chunk_overlap,
                    "processed_data": processed_result
                }
                all_processed_results.append(result_with_metadata)
            except Exception as e:
                print(
                    f"  Error processing chunk {i+1} of {file_name}: {str(e)}")

    # Save results if output path specified
    if output_file_path:
        save_processed_results(all_processed_results, output_file_path)

    print(
        f"\nProcessing complete. Processed {len(all_processed_results)} chunks from {len(files_content)} files.")
    return all_processed_results


def save_processed_results(results: List[Dict[str, Any]], output_file_path: str) -> None:
    """Save processed results to various file formats.

    Args:
        results: List of processed results with metadata
        output_file_path: Path to save the results
    """
    print(f"Saving results to: {output_file_path}")
    file_ext = os.path.splitext(output_file_path)[1].lower()

    if file_ext == '.json':
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    elif file_ext == '.jsonl':
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    elif file_ext in ['.txt', '.csv', '.tsv']:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(str(result["processed_data"]) + '\n')
    else:
        # Default to JSON
        output_file_path = output_file_path + '.json'
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file_path}")


def prepare_for_vector_db(chunks: List[str], source_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Prepare text chunks for vector database ingestion.

    Args:
        chunks: List of text chunks
        source_info: Optional metadata about the source

    Returns:
        List of dictionaries ready for vector database
    """
    prepared_data = []

    for i, chunk in enumerate(chunks):
        metadata = {
            'chunk_index': i,
            'chunk_size': len(chunk),
            'timestamp': datetime.now().isoformat()
        }

        if source_info:
            metadata.update(source_info)

        prepared_data.append({
            'text': chunk,
            'metadata': metadata
        })

    return prepared_data


# if __name__ == "__main__":
#     # Example usage
#     directory_path = "Data"
#     results = process_directory(
#         directory_path=directory_path,
#         chunk_method="auto",
#         chunk_size=1000,
#         output_file_path="processed_output.json"
#     )
#     print(f"Processed {len(results)} chunks total")
