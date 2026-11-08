"""
Quick start WITHOUT a checkpoint: NEXUS-16 builds and encodes from random init.

The pretrained checkpoint (models/nexus16_v15_100m.pt) improves quality,
but the architecture itself runs with zero setup.
"""
from nexus16 import NEXUS16

model = NEXUS16()

emb = model.encode("The quick brown fox")
print(f"Embedding shape: {tuple(emb.shape)}")  # (256,)

sim = model.similarity(
    "The cat is sleeping on the mat",
    "A kitten is resting on the rug",
)
print(f"Similarity (random init, untrained): {sim:.4f}")
print("Note: for real semantic quality, download the pretrained checkpoint (README).")
