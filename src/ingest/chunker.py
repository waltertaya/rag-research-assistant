from typing import List

import tiktoken


def chunk_text_by_tokens(text: str, chunk_size: int = 512, overlap: int = 64, encoding_name: str = "cl100k_base") -> List[str]:
    """Chunk text into smaller pieces based on token count.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum number of tokens per chunk.
        overlap (int): The number of overlapping tokens between chunks.
        encoding_name (str): The name of the token encoding to use.

    Returns:
        List[str]: A list of text chunks.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    text_length = len(tokens)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end == text_length:
            break
        
        start += chunk_size - overlap
    
    return chunks