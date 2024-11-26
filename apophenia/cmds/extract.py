import argparse
import importlib
import logging
import os
import sys
import textwrap

from apophenia.api.extract import process_repository

logger = logging.getLogger(__name__)

command_description = """
Extract and structure data from a Git repository for RAG.
"""
long_description = """
The 'extract' command analyzes the given git repository and transform it
to become usable in a RAG context.
"""


def add_arguments(parser):
    """
    Adds the argument options to the extract command parser.

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments are added.
    """
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = long_description

    parser.add_argument(
        "repo_url", type=str, help="URL of the Git repository to clone."
    )
    parser.add_argument(
        "--clone_dir",
        type=str,
        default="./local_repo",
        help="Directory to clone the repository into.",
    )
    parser.add_argument(
        "--faiss_path",
        type=str,
        default="repository_index.faiss",
        help="Path to save the FAISS index.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="repository_metadata.json",
        help="Path to save the metadata JSON.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Size of content chunks (in characters).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the SentenceTransformer model to use.",
    )


def main(args):
    process_repository(
        repo_url=args.repo_url,
        clone_dir=args.clone_dir,
        faiss_path=args.faiss_path,
        metadata_path=args.metadata_path,
        chunk_size=args.chunk_size,
        model_name=args.model_name,
    )
