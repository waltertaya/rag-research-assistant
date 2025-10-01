import click
import json
from pathlib import Path
from src.utils import config
from src.ingest.parser import parse_file
from src.ingest.chunker import chunk_text_by_tokens
from src.embeddings.client import embed_texts
from src.index.indexer import FaissIndexer
from src.retriever.retriever import Retriever
from src.prompt.prompt import build_prompt

from openai import OpenAI

client = OpenAI()


@click.group()
def cli():
    pass


@cli.command()
@click.argument("path")
def ingest(path):
    """Ingest a single file (pdf/txt)."""
    print("Parsing file", path)
    text, meta = parse_file(path)
    chunks = chunk_text_by_tokens(text)
    print(f"Generated {len(chunks)} chunks. Generating embeddings...")
    vecs = embed_texts(chunks)
    metas = []
    for idx, (c_text, v) in enumerate(zip(chunks, vecs)):
        m = {
            "chunk_id": idx,
            "file_name": meta.get("source"),
            "text": c_text,
            "vector": v.tolist(),
        }
        metas.append(m)

    idx = FaissIndexer.load()
    if idx is None:
        idx = FaissIndexer(dim=len(vecs[0]))
    idx.add(vecs, metas)
    idx.save()
    print("Ingestion complete and index saved.")


@cli.command()
@click.argument("query")
@click.option("--top-k", default=5)
def query(query, top_k):
    print("Loading index and running retrieval...")
    idx = FaissIndexer.load()
    if idx is None:
        print("Index not found. Ingest documents first.")
        return
    r = Retriever(idx)
    results = r.retrieve(query, top_k=top_k)

    prompt = build_prompt(query, results)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
    )
    out = resp.choices[0].message.content
    print("\n=== ANSWER ===\n")
    print(out)
    print("\n=== SOURCES (retrieved) ===")
    for r in results:
        m = r.get("metadata", {})
        print(f"- {m.get('file_name')} :: {m.get('chunk_id')} (score={r['score']:.3f})")


if __name__ == "__main__":
    cli()
