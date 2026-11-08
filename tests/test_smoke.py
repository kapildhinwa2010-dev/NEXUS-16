import torch

from nexus16 import NEXUS16


def test_build_and_encode_tiny():
    """Bounded smoke test: the architecture builds and encodes without any checkpoint."""
    model = NEXUS16(d_model=32, d_wave=16, n_mamba=1)
    model.eval()
    e1 = model.encode("hello world")
    e2 = model.encode("hello world")
    e3 = model.encode("something else entirely")
    assert tuple(e1.shape) == (32,)
    assert torch.allclose(e1, e2), "same text must give the same embedding"
    assert not torch.allclose(e1, e3), "different text must give a different embedding"


def test_forward_batch():
    model = NEXUS16(d_model=32, d_wave=16, n_mamba=1)
    model.eval()
    x = torch.randint(0, 256, (2, 10))
    out = model(x)
    assert tuple(out.shape) == (2, 32)
