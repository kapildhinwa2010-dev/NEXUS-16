"""
Example: Semantic Search with NEXUS-16
"""
from nexus16 import NEXUS16

# Load model
model = NEXUS16.from_pretrained("models/nexus16_v15_100m.pt")

# Document database
documents = [
    "The cat sleeps peacefully on the rug",
    "Quantum physics explains subatomic particles",
    "Pizza with cheese is absolutely delicious",
    "The kitten is resting on the mat",
    "The car drives fast on the highway",
    "Water boils at 100 degrees Celsius",
]

# Search query
query = "kitten resting"

print(f"Query: '{query}'")
print("\nTop 3 Results:")
results = model.search(query, documents, top_k=3)
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc}")