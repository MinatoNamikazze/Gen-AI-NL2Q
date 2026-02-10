import chromadb
import pandas as pd
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

CSV_PATH = "test.csv"

class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input)
        return [embedding.tolist() for embedding in embeddings]

def setup_vector_db():
    client = chromadb.PersistentClient(path="./chroma_storage")

    try:
        client.delete_collection(name="matrix_quotes")
    except:
        pass

    matrix_quotes = client.create_collection(
        name="matrix_quotes",
        embedding_function=CustomEmbeddingFunction()
    )

    df = pd.read_csv(CSV_PATH)
    documents, metadata, ids = [], [], []

    for index, row in df.iterrows():
        # Concatenate prompt + schema as the embedding input
        embedding_input = f"{row['sql_prompt']} | Schema: {row['sql_context']}"
        documents.append(embedding_input)
        ids.append(str(index))
        metadata.append({
            "sql": row["sql"],
            "explanation": row["sql_explanation"],
            "context": row["sql_context"],
            "prompt": row["sql_prompt"]
        })

    BATCH_SIZE = 5000
    for i in range(0, len(documents), BATCH_SIZE):
        matrix_quotes.add(
            documents=documents[i:i + BATCH_SIZE],
            metadatas=metadata[i:i + BATCH_SIZE],
            ids=ids[i:i + BATCH_SIZE]
        )

    return matrix_quotes

def retrieve_similar_docs(matrix_quotes, question: str, schema: str):
    combined_query = f"{question} | Schema: {schema}"
    results = matrix_quotes.query(
        query_texts=[combined_query],
        n_results=2
    )
    return results
