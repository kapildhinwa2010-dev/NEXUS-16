"""
Semantic similarity demo - bina checkpoint ke bhi chalta hai.
(Random init pe similarities meaningful nahi hote - asli quality
ke liye pretrained checkpoint download karo, README dekho.)
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nexus16 import NEXUS16

model = NEXUS16()

pairs = [
    ("The cat is sleeping on the mat", "A kitten is resting on the rug"),
    ("Quantum physics explains reality", "Pizza is a delicious food"),
]
for a, b in pairs:
    print(f"sim={model.similarity(a, b):.4f}  {a!r} <-> {b!r}")
