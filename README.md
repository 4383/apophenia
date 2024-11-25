# apophenia

![Build](https://github.com/4383/apophenia/actions/workflows/main.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/apophenia.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/apophenia.svg)
![PyPI - Status](https://img.shields.io/pypi/status/apophenia.svg)
[![Downloads](https://pepy.tech/badge/apophenia)](https://pepy.tech/project/apophenia)
[![Downloads](https://pepy.tech/badge/apophenia/month)](https://pepy.tech/project/apophenia/month)

Apophenia give meaning to any existing Git repository.

Apophenia extract and structure all the data from a Git repository to make
them usable in RAG or in with AI agents.

Apophenia impose a meaningful interpretation on a nebulous stimulus (a Git
repo).

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

## Why using apophenia?

Here’s a **list of potential applications** for using the generated results
(FAISS vectors and JSON metadata) within a **RAG
(Retrieval-Augmented Generation)** system:

### **1. Augmented Documentation**
- Generate enriched answers by combining documentation, commit messages, and code.
- Example questions:
  - *"How do I use the `authenticate_user` function?"*
  - *"What is the structure of this project?"*

### **2. Developer Assistance**
- Quickly search for specific parts of the code or documentation.
- Identify relevant functions or files based on queries like:
  - *"Where is the authentication logic implemented?"*
  - *"Which module handles network connections?"*

### **3. Intelligent Debugging**
- Retrieve historical information to understand bugs or errors.
- Analyze recent changes with queries like:
  - *"What are the latest modifications in this file?"*
  - *"Which commits mention this bug?"*

### **4. Automated Changelog Generation**
- Create a changelog based on commit messages and their diffs.
- Example use case:
  - Automatically generate a structured changelog for a new release.

### **5. Code Migration and Modernization**
- Identify outdated dependencies or technologies.
- Plan migrations by answering queries like:
  - *"Which files are using Eventlet?"*
  - *"Which commits introduced asyncio?"*

### **6. Audit and Compliance**
- Search for changes related to vulnerabilities or critical dependencies.
- Example questions:
  - *"Which files use OpenSSL?"*
  - *"Which commits fixed vulnerabilities?"*

### **7. Documentation Generation**
- Generate guides or technical manuals from existing code and documentation fragments.
- Example:
  - Create an installation guide from README files and configuration scripts.

### **8. Git History Analysis**
- Understand individual contributions or file evolution.
- Example questions:
  - *"Who wrote this function?"*
  - *"What are John Doe's contributions?"*

### **9. Keyword or Contextual Search**
- Search for specific concepts within the project:
  - *"Where is the caching logic handled?"*
  - *"Which files mention secure connections?"*

### **10. Onboarding Assistance**
- Simplify onboarding for new developers:
  - Provide guided answers like:
    - *"The main features of this project are documented in `README.md`."*
    - *"`auth.py` handles authentication logic."*

### **11. Change Impact Analysis**
- Identify which files or functions are impacted by a specific commit.
- Example questions:
  - *"Which files were modified by this commit?"*
  - *"Which tests are affected by this change?"*

### **12. Code Example Generation**
- Extract code examples from existing fragments in files or commits.
- Example use case:
  - Generate a snippet to illustrate how to use a specific function or module.

### **13. Technical Problem Solving**
- Quickly find useful information to solve a technical issue.
- Example questions:
  - *"Which file is responsible for this exception?"*
  - *"Which commit introduced this error?"*

### **14. Dependency Auditing**
- Identify the libraries used and their versions.
- Example questions:
  - *"Which version of Django is being used?"*
  - *"Which commits mention outdated dependencies?"*

### **15. Performance Analysis**
- Search for changes related to performance optimization.
- Example questions:
  - *"Which commits optimized this file?"*
  - *"Which functions were refactored for better performance?"*

### **16. Contribution Analysis for Project Management**
- Identify team members who are most active in certain areas of the project.
- Example questions:
  - *"Who contributes the most to the networking module?"*
  - *"What are the primary files in this project?"*

### **17. Technical Report Generation**
- Create customized reports on the state or evolution of a project.
- Examples:
  - Report on the 10 most significant recent commits.
  - List of main modules and the most modified files.

### **18. Automating DevOps Workflows**
- Integrate extracted data into CI/CD pipelines.
- Example:
  - Identify critical files for a specific build task.

### **19. Comparative Analysis**
- Compare versions of files or branches using diffs and commits.

### **20. Continuous Improvement**
- Identify areas of the code that need documentation or refactoring.
- Example questions:
  - *"Which files lack associated documentation?"*
  - *"Which commits mention suboptimal code?"*

If you recognize yourself in one of these examples then Apophenia is for you.

## Going Further with FAISS

You can use generated output FAISS with [langchain](
https://python.langchain.com/docs/integrations/vectorstores/faiss/)
or with any modern libraries like [llamaindex](
https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/faiss/)

- https://github.com/facebookresearch/faiss
- https://pypi.org/project/faiss/
