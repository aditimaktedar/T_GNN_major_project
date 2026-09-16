import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

METADATA_PATH = os.path.join(
    ROOT,
    "data/rag/embeddings/rag_metadata.csv"
)

INDEX_PATH = os.path.join(
    ROOT,
    "data/rag/embeddings/rag.index"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"


print("=" * 80)
print("DDI RAG RETRIEVAL TEST")
print("=" * 80)


# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

print("\nLoading metadata...")

metadata = pd.read_csv(METADATA_PATH)

metadata["drugbank_id"] = (
    metadata["drugbank_id"]
    .astype(str)
    .str.strip()
)

metadata["vector_index"] = (
    metadata["vector_index"]
    .astype(int)
)

print("Metadata shape:", metadata.shape)
print(
    "Unique DrugBank IDs:",
    metadata["drugbank_id"].nunique()
)


# ------------------------------------------------------------
# Load FAISS
# ------------------------------------------------------------

print("\nLoading FAISS index...")

index = faiss.read_index(INDEX_PATH)

print("FAISS vectors:", index.ntotal)
print("FAISS dimension:", index.d)


assert len(metadata) == index.ntotal
assert index.d == 384

print("Index/metadata consistency: PASS")


# ------------------------------------------------------------
# Load embedding model
# ------------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model:", MODEL_NAME)


# ------------------------------------------------------------
# Target-specific retrieval
# ------------------------------------------------------------

def retrieve_target_drug_evidence(
    query,
    target_drugbank_id,
    top_k=10
):

    target_drugbank_id = str(
        target_drugbank_id
    ).strip()

    target_rows = metadata[
        metadata["drugbank_id"]
        == target_drugbank_id
    ]

    if target_rows.empty:
        return []

    vector_indices = (
        target_rows["vector_index"]
        .astype(int)
        .tolist()
    )

    vectors = np.vstack([
        index.reconstruct(idx)
        for idx in vector_indices
    ]).astype("float32")

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores = np.dot(
        vectors,
        query_vector[0]
    )

    order = np.argsort(scores)[::-1][:top_k]

    results = []

    rows = target_rows.iloc[order]

    for rank, (position, (_, row)) in enumerate(
        zip(order, rows.iterrows()),
        start=1
    ):

        results.append({
            "rank": rank,
            "score": float(scores[position]),
            "drugbank_id": row["drugbank_id"],
            "chunk_id": row["chunk_id"],
            "source": row["source"],
            "section": row["section"],
            "text": row["text"]
        })

    return results


# ------------------------------------------------------------
# TEST DRUG
# ------------------------------------------------------------

TEST_DRUG = "DB00213"
TEST_NAME = "Pantoprazole"

QUERY = (
    "Pantoprazole mechanism of action "
    "pharmacology proton pump"
)

print("\n" + "=" * 80)
print("RETRIEVAL TEST")
print("=" * 80)

print("Drug:", TEST_NAME)
print("DrugBank ID:", TEST_DRUG)
print("Query:", QUERY)


results = retrieve_target_drug_evidence(
    QUERY,
    TEST_DRUG,
    top_k=5
)


print("\nRetrieved results:", len(results))


for result in results:

    print("\n" + "-" * 80)

    print("Rank:", result["rank"])
    print("Score:", round(result["score"], 4))
    print("DrugBank:", result["drugbank_id"])
    print("Chunk:", result["chunk_id"])
    print("Source:", result["source"])
    print("Section:", result["section"])

    print("\nText:")
    print(result["text"][:1000])


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL TEST")
print("=" * 80)

assert len(results) > 0

assert all(
    r["drugbank_id"] == TEST_DRUG
    for r in results
)

print("Target-specific retrieval: PASS")
print("All retrieved chunks belong to:", TEST_DRUG)
print("\nRAG RETRIEVAL TEST PASSED")
