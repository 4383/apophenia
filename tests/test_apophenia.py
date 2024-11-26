import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

import faiss
import numpy as np
from git import Repo

from apophenia.api.extract import (
    clone_repo,
    extract_commit_history,
    generate_embeddings,
    process_repository,
    save_to_faiss_and_json,
    segment_content,
)


class TestGitProcessing(unittest.TestCase):
    def test_segment_content(self):
        # Test content segmentation
        content = "abcdef" * 100  # 600 characters
        chunk_size = 100
        chunks = segment_content(content, chunk_size)
        self.assertEqual(len(chunks), 6)
        self.assertTrue(all(len(chunk) == chunk_size for chunk in chunks))

    @patch("apophenia.api.extract.Repo")
    def test_extract_commit_history(self, mock_repo):
        # Mock commit history
        mock_commit = MagicMock()
        mock_commit.message = "Initial commit"
        mock_commit.hexsha = "123abc"
        mock_commit.author.name = "Author"
        mock_commit.committed_datetime.isoformat.return_value = "2024-01-01T12:00:00"
        mock_commit.diff.return_value = ["diff --git a/file b/file"]
        mock_repo.iter_commits.return_value = [mock_commit]

        commit_data = extract_commit_history(mock_repo)
        self.assertEqual(len(commit_data), 1)
        self.assertEqual(commit_data[0]["message"], "Initial commit")
        self.assertIn("diff --git", commit_data[0]["diff"])

    @patch("apophenia.api.extract.SentenceTransformer")
    def test_generate_embeddings(self, mock_model):
        # Mock embedding generation
        fragments = ["test fragment 1", "test fragment 2"]

        # Mock the behavior of encode to return a tensor-like object
        class MockTensor:
            def cpu(self):
                return self

            def numpy(self):
                return np.random.rand(384)

        mock_model_instance = mock_model.return_value
        mock_model_instance.encode.side_effect = (
            lambda x, convert_to_tensor: MockTensor()
        )

        # Call the function and verify the results
        embeddings = generate_embeddings(fragments, mock_model_instance)
        self.assertEqual(len(embeddings), len(fragments))
        self.assertEqual(len(embeddings[0]), 384)

    def test_save_to_faiss_and_json(self):
        # Test FAISS and JSON saving
        embeddings = [np.random.rand(384) for _ in range(5)]
        metadata = [{"key": "value"} for _ in range(5)]

        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_path = os.path.join(temp_dir, "index.faiss")
            metadata_path = os.path.join(temp_dir, "metadata.json")

            save_to_faiss_and_json(embeddings, metadata, faiss_path, metadata_path)

            # Check FAISS file
            self.assertTrue(os.path.exists(faiss_path))
            index = faiss.read_index(faiss_path)
            self.assertEqual(index.ntotal, len(embeddings))

            # Check JSON file
            self.assertTrue(os.path.exists(metadata_path))
            with open(metadata_path, "r") as f:
                loaded_metadata = json.load(f)
                self.assertEqual(loaded_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
