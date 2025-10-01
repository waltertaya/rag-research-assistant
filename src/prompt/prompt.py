PROMPT_TEMPLATE = """
You are an assistant that answers questions using only the provided context. If the answer is not in the context, say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer concisely.
- After the answer, add a SOURCES section listing file_name and chunk_id for the pieces used.
"""


def build_prompt(question: str, retrieved: list) -> str:
    ''' Prompt builder for LLMs.'''
    context_parts = []
    for item in retrieved:
        m = item.get("metadata", {})
        snippet = m.get("text", "")[:1000]
        context_parts.append(f"FILE: {m.get('file_name')} | CHUNK: {m.get('chunk_id')}\n{snippet}")
    
    context = "\n\n---\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return prompt
