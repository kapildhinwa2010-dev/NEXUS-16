"""
Example: Document Clustering with NEXUS-16
"""
from nexus16 import NEXUS16
import torch

# Load model
model = NEXUS16.from_pretrained("models/nexus16_v15_100m.pt")

# Documents to cluster
documents = [
    "The cat is sleeping",
    "A kitten is resting",
    "Pizza is delicious",
    "Food tastes amazing",
    "Quantum physics is hard",
    "Science is fascinating",
]

# Get embeddings
embeddings = []
for doc in documents:
    emb = model.encode(doc)
    embeddings.append(emb)

# Stack embeddings
emb_matrix = torch.stack(embeddings)

# Simple clustering (cosine similarity threshold)
print("Document Clusters:")
print("-" * 60)
threshold = 0.4
clusters = []
used = set()

for i in range(len(documents)):
    if i in used:
        continue
    cluster = [documents[i]]
    used.add(i)
    for j in range(i+1, len(documents)):
        if j in used:
            continue
        sim = torch.cosine_similarity(emb_matrix[i], emb_matrix[j]).item()
        if sim > threshold:
            cluster.append(documents[j])
            used.add(j)
    clusters.append(cluster)

for i, cluster in enumerate(clusters, 1):
    print(f"Cluster {i}:")
    for doc in cluster:
        print(f"  - {doc}")
    print()