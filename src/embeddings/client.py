import os
import json
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
import numpy as np
from src.utils import config

client = OpenAI()
EMBED_CACHE = Path(config.DATA_DIR) / "embeddings_cache.json"


def _load_cache():
    if EMBED_CACHE.exists():
        return json.loads(EMBED_CACHE.read_text())
    return {}


def _save_cache(cache: dict):
    EMBED_CACHE.write_text(json.dumps(cache))


def embed_texts(texts: List[str], model: str = None) -> List[List[float]]:
    '''Embeddings helper using OpenAI API, with simple caching.
    '''
    model = model or config.EMBEDDING_MODEL
    cache = _load_cache()
    results = []
    to_call = []
    to_call_idx = []

    for i, t in enumerate(texts):
        key = str(hash(t))
        if key in cache:
            results.append(cache[key])
        else:
            results.append(None)
            to_call.append(t)
            to_call_idx.append(i)
    
    if to_call:
        response = client.embeddings.create(
            input=to_call,
            model=model,
        )
        for idx, item in enumerate(response.data):
            vector = item.embedding
            i = to_call_idx[idx]
            results[i] = vector
            key = str(hash(to_call[idx]))
            cache[key] = vector
        _save_cache(cache)

    results_np = [np.array(r, dtype=float) for r in results]
    return results_np
