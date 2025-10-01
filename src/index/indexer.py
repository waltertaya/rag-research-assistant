from pathlib import Path
import faiss
import numpy as np
import pickle
from typing import List, Tuple, Dict
from src.utils import config


INDEX_PATH = Path(config.INDEX_DIR) / "faiss.index"
METADATA_PATH = Path(config.INDEX_DIR) / "metadata.pkl"


class FaissIndexer:
    ''' FAISS index wrapper: create, persist, load, search. '''

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

    
    def add(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        ''' Add vectors and their metadata to the index. '''
        if not vectors:
            return
        vecs_np = np.vstack([v / np.linalg.norm(v) for v in vectors]).astype('float32')
        # faiss.normalize_L2(vecs_np)
        self.index.add(vecs_np)
        self.metadata.extend(metadatas)


    def save(self):
        ''' Save the index and metadata to disk. '''
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)
    

    @classmethod
    def load(cls):
        ''' Load the index and metadata from disk. '''
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            return None
        
        index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        
        dim = len(metadata[0]['vector']) if metadata else 1536
        indexer = cls(dim)
        indexer.index = index
        indexer.metadata = metadata
        return indexer
    

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        ''' Search the index for the top_k most similar vectors to the query_vector. '''
        if self.index.ntotal == 0:
            return []
        
        query_np = (query_vector / np.linalg.norm(query_vector)).astype('float32').reshape(1, -1)
        scores, ids = self.index.search(query_np, top_k)
        
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < len(self.metadata):
                results.append({"metadata": self.metadata[idx], "score": float(score)})
        
        return results
