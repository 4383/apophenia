import argparse
import json
import os

import faiss
import numpy as np
from git import Repo
from sentence_transformers import SentenceTransformer


def clone_repo(repo_url, clone_dir):
    if not os.path.exists(clone_dir):
        print(f"Cloning repository from {repo_url}...")
        Repo.clone_from(repo_url, clone_dir)
    else:
        print(f"Repository already exists at {clone_dir}.")
    return Repo(clone_dir)


def segment_content(content, chunk_size=500):
    """Splits content into smaller chunks."""
    return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]


def extract_commit_history(repo):
    """
    Extract commit history: messages, diffs, and metadata.
    """
    commit_data = []
    for commit in repo.iter_commits():
        commit_message = commit.message.strip()
        commit_diff = commit.diff(None, create_patch=True)  # Diff patch
        diff_text = "\n".join(str(diff) for diff in commit_diff)

        commit_data.append(
            {
                "hash": commit.hexsha,
                "author": commit.author.name,
                "date": commit.committed_datetime.isoformat(),
                "message": commit_message,
                "diff": diff_text,
            }
        )
    return commit_data


def generate_embeddings(fragments, model):
    """
    Generate embeddings for each fragment.
    """
    embeddings = []
    for fragment in fragments:
        embedding = model.encode(fragment, convert_to_tensor=True).cpu().numpy()
        embeddings.append(embedding)
    return embeddings


def save_to_faiss_and_json(embeddings, metadata, faiss_path, metadata_path):
    """
    Save embeddings to FAISS and metadata to JSON.
    """
    # Save embeddings to FAISS
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, faiss_path)

    # Save metadata to JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def process_repository(
    repo_url, clone_dir, faiss_path, metadata_path, chunk_size, model_name
):
    repo = clone_repo(repo_url, clone_dir)
    embedding_model = SentenceTransformer(model_name)

    all_embeddings = []
    all_metadata = []

    # Process file content
    for root, _, files in os.walk(clone_dir):
        if ".git" in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                fragments = segment_content(content, chunk_size)
                embeddings = generate_embeddings(fragments, embedding_model)

                for fragment, embedding in zip(fragments, embeddings):
                    all_metadata.append(
                        {
                            "type": "file",
                            "file_path": file_path,
                            "content_preview": fragment[:50],  # Preview for context
                            "metadata": {
                                "lines": len(content.splitlines()),
                                "size": len(content),
                            },
                        }
                    )
                    all_embeddings.append(embedding)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

    # Process commit history
    commit_data = extract_commit_history(repo)
    for commit in commit_data:
        message = commit["message"]
        diff = commit["diff"]

        # Embed commit message
        if message:
            message_embedding = (
                embedding_model.encode(message, convert_to_tensor=True).cpu().numpy()
            )
            all_embeddings.append(message_embedding)
            all_metadata.append(
                {
                    "type": "commit",
                    "hash": commit["hash"],
                    "author": commit["author"],
                    "date": commit["date"],
                    "content_preview": message[:50],  # Preview
                }
            )

        # Embed commit diff (if not too large)
        if diff and len(diff) < 1000:  # Avoid embedding overly large diffs
            diff_embedding = (
                embedding_model.encode(diff, convert_to_tensor=True).cpu().numpy()
            )
            all_embeddings.append(diff_embedding)
            all_metadata.append(
                {
                    "type": "diff",
                    "hash": commit["hash"],
                    "author": commit["author"],
                    "date": commit["date"],
                    "content_preview": diff[:50],  # Preview
                }
            )

    # Save embeddings and metadata
    save_to_faiss_and_json(all_embeddings, all_metadata, faiss_path, metadata_path)

    print(f"FAISS index saved to {faiss_path}")
    print(f"Metadata saved to {metadata_path}")


def main():
    print("starting apophenia")
    parser = argparse.ArgumentParser(
        description="Extract and structure data from a Git repository for RAG."
    )
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

    args = parser.parse_args()

    process_repository(
        repo_url=args.repo_url,
        clone_dir=args.clone_dir,
        faiss_path=args.faiss_path,
        metadata_path=args.metadata_path,
        chunk_size=args.chunk_size,
        model_name=args.model_name,
    )


# Argparse configuration
if __name__ == "__main__":
    main()
