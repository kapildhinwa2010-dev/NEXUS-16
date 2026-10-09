"""
Example: Semantic Similarity with NEXUS-16
"""
from nexus16 import NEXUS16

# Load model
model = NEXUS16.from_pretrained("models/nexus16_v15_100m.pt")

# Test pairs
pairs = [
    ("The cat is sleeping on the mat", "A kitten is resting on the rug"),
    ("She cooked dinner for the family", "He prepared a meal at home"),
    ("Quantum physics is complex", "Pizza is delicious"),
]

print("Semantic Similarity Results:")
print("-" * 60)
for text1, text2 in pairs:
    sim = model.similarity(text1, text2)
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {sim:.4f}")
    print("-" * 60)