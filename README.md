# apophenia

![Build](https://github.com/4383/apophenia/actions/workflows/main.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/apophenia.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/apophenia.svg)
![PyPI - Status](https://img.shields.io/pypi/status/apophenia.svg)
[![Downloads](https://pepy.tech/badge/apophenia)](https://pepy.tech/project/apophenia)
[![Downloads](https://pepy.tech/badge/apophenia/month)](https://pepy.tech/project/apophenia/month)

Impose a meaningful interpretation on a nebulous stimulus (a Git repo).

Extract and structure all the data from a Git repository to make them usable
in RAG or in with AI agents.

## Usage

Extract data from a given repository:

```bash
$ pip install apophenia
$ apophenia https://github.com/4383/niet \
  --faiss_path /tmp/results.faiss \
  --metadata_path /tmp/results.json
```

And use generated data in a RAG (python snippet example):

```python
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the FAISS index and the JSON metadata previously generated
def load_index_and_metadata(faiss_path, metadata_path):
    index = faiss.read_index(faiss_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

# Embedding of the user request
def embed_query(query, model):
    return model.encode(query, convert_to_tensor=True).cpu().numpy()

# Seach in the FAISS index
def search_in_faiss(index, query_embedding, metadata, k=5):
    distances, indices = index.search(np.array([query_embedding]), k)
    results = []
    for i, idx in enumerate(indices[0]):
        result = metadata[idx]
        result['distance'] = distances[0][i]
        results.append(result)
    return results

# Build a prompt for a generative model
def build_prompt(query, retrieved_info):
    prompt = f"Answer the following question based on the retrieved information:\n\n"
    prompt += f"Question: {query}\n\n"
    prompt += "Retrieved Information:\n"
    for info in retrieved_info:
        content_type = info.get("type", "unknown")
        content_preview = info.get("content_preview", "No preview available")
        prompt += f"- {content_type.upper()}: {content_preview}\n"
    prompt += "\nYour Answer:"
    return prompt

# Generate a response with a generative model
def generate_response(prompt, model_name="EleutherAI/gpt-neo-125M", max_length=200):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def run_rag_system(query, faiss_path, metadata_path, embedding_model_name, generative_model_name):
    # Load data (FAISS index and metadata, and embedding)
    index, metadata = load_index_and_metadata(faiss_path, metadata_path)
    embedding_model = SentenceTransformer(embedding_model_name)

    query_embedding = embed_query(query, embedding_model)

    # Search in FAISS
    retrieved_info = search_in_faiss(index, query_embedding, metadata)

    prompt = build_prompt(query, retrieved_info)

    response = generate_response(prompt, model_name=generative_model_name)

    return response, retrieved_info

if __name__ == "__main__":
    # Configuration
    FAISS_PATH = "results.faiss"
    METADATA_PATH = "results.json"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    GENERATIVE_MODEL_NAME = "EleutherAI/gpt-neo-125M"

    query = "How does the authentication system work in this repository?"

    response, retrieved_info = run_rag_system(
        query=query,
        faiss_path=FAISS_PATH,
        metadata_path=METADATA_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        generative_model_name=GENERATIVE_MODEL_NAME
    )

    print("Generated Response:")
    print(response)
    print("\nRetrieved Information:")
    for info in retrieved_info:
        print(info)
```

## About FAISS

You can use generated output FAISS with [langchain](
https://python.langchain.com/docs/integrations/vectorstores/faiss/)
or with any modern libraries like [llamaindex](
https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/faiss/)

- https://github.com/facebookresearch/faiss
- https://pypi.org/project/faiss/
