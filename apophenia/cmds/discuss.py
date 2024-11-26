import argparse
import importlib
import logging
import os
import sys
import tempfile

from apophenia.api.discuss import discuss
from apophenia.api.extract import process_repository

logger = logging.getLogger(__name__)

command_description = """
Discuss with your git repository and your data.
"""
long_description = """
Discuss with your git repository and your data.
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
        "--chunk_size",
        type=int,
        default=500,
        help="Size of content chunks (in characters).",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the SentenceTransformer model to use.",
    )
    parser.add_argument(
        "--generative_model_name",
        type=str,
        default="EleutherAI/gpt-neo-125M",
        help="Name of the generative model to use.",
    )


def main(args):
    tmp_dir = tempfile.mktemp()
    faiss_file = os.path.join(tmp_dir, "result.faiss")
    metadata_file = os.path.join(tmp_dir, "result.json")
    process_repository(
        repo_url=args.repo_url,
        clone_dir=tmp_dir,
        faiss_path=faiss_file,
        metadata_path=metadata_file,
        chunk_size=args.chunk_size,
        model_name=args.embedding_model_name,
    )

    discuss(
        faiss_file,
        metadata_file,
        args.embedding_model_name,
        args.generative_model_name,
    )
