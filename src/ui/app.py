import sys
from pathlib import Path
import os
import streamlit as st


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.parser import parse_file
from src.ingest.chunker import chunk_text_by_tokens
from src.embeddings.client import embed_texts
from src.index.indexer import FaissIndexer
from src.retriever.retriever import Retriever
from src.prompt.prompt import build_prompt
from src.utils import config
from openai import OpenAI
client = OpenAI()

st.title("RAG Research Assistant — Starter")

uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt", "md", "docx", "csv"])
if uploaded is not None:
    path = f"{config.DATA_DIR}/{uploaded.name}"
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success("Saved file. Run ingest to add to index.")
    if st.button("Ingest now"):
        text, meta = parse_file(path)
        chunks = chunk_text_by_tokens(text)
        vecs = embed_texts(chunks)
        metas = []
        for idx, (c_text, v) in enumerate(zip(chunks, vecs)):
            m = {
                "chunk_id": idx,
                "file_name": meta.get("source", os.path.basename(path)),
                "text": c_text,
                "vector": v.tolist(),
            }
            metas.append(m)
        idx = FaissIndexer.load()
        if idx is None:
            idx = FaissIndexer(dim=len(vecs[0]))
        idx.add(vecs, metas)
        idx.save()
        st.success("Ingested and indexed.")

query = st.text_input("Ask a question about your uploaded documents")
if st.button("Search") and query:
    idx = FaissIndexer.load()
    if idx is None:
        st.warning("Index not found — ingest some documents first.")
    else:
        r = Retriever(idx)
        results = r.retrieve(query, top_k=5)
        prompt = build_prompt(query, results)
        with st.spinner("Querying LLM..."):
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.0,
            )
        out = resp.choices[0].message.content
        st.subheader("Answer")
        st.write(out)
        st.subheader("Sources")
        for r_item in results:
            m = r_item.get("metadata", {})
            st.write(f"- {m.get('file_name')} :: {m.get('chunk_id')} (score={r_item['score']:.3f})")
