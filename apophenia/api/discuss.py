import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# Load the FAISS index and the JSON metadata previously generated
def load_index_and_metadata(faiss_path, metadata_path):
    index = faiss.read_index(faiss_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
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
        result["distance"] = distances[0][i]
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


def run_rag_system(
    query, faiss_path, metadata_path, embedding_model_name, generative_model_name
):
    # Load data (FAISS index and metadata, and embedding)
    index, metadata = load_index_and_metadata(faiss_path, metadata_path)
    embedding_model = SentenceTransformer(embedding_model_name)
    query_embedding = embed_query(query, embedding_model)

    # Search in FAISS
    retrieved_info = search_in_faiss(index, query_embedding, metadata)
    prompt = build_prompt(query, retrieved_info)
    response = generate_response(prompt, model_name=generative_model_name)

    return response, retrieved_info


def discuss(faiss_file, metadata_file, embedding_model, generative_model):

    while True:
        try:
            query = input("ask your question> ")

            response, retrieved_info = run_rag_system(
                query=query,
                faiss_path=faiss_file,
                metadata_path=metadata_file,
                embedding_model_name=embedding_model,
                generative_model_name=generative_model,
            )

            print("Generated Response:")
            print(response)
            print("\nRetrieved Information:")
            for info in retrieved_info:
                print(info)
        except KeyboardInterrupt:
            print("Bye bye!")
            return
