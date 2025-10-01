from typing import List, Dict
import numpy as np
from src.embeddings.client import embed_texts
from src.index.indexer import FaissIndexer
from src.utils import config


class Retriever:
    ''' Retriever class to handle document retrieval using FAISS and OpenAI embeddings. '''

    def __init__(self, indexer: FaissIndexer):
        self.indexer = indexer


    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        ''' Retrieve top_k most relevant documents for the given query. '''
        query_vector = embed_texts([query])[0]
        return self.indexer.search(query_vector, top_k)
