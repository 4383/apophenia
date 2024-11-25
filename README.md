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

```bash
$ pip install apophenia
$ apophenia https://github.com/4383/niet \
  --faiss_path /tmp/faiss_result \
  --metadata_path /tmp/json_result
```

## About FAISS

You can use generated output FAISS with [langchain](
https://python.langchain.com/docs/integrations/vectorstores/faiss/)
or with any modern libraries like [llamaindex](
https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/faiss/)

- https://github.com/facebookresearch/faiss
- https://pypi.org/project/faiss/
