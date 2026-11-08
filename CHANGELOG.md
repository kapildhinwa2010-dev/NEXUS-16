# Changelog

## [0.1.0] - 2026-11-08

### Added
- Single-file NEXUS-16 architecture: WaveFeatures, SpiralRoPE, MambaBlock,
  AttentionPool - raw bytes -> 256-dim semantic embedding.
- from_pretrained: clear FileNotFoundError + gdown instructions (missing checkpoint pe
  ugly crash tha).
- No-checkpoint quickstart (examples/quickstart_no_checkpoint.py) +
  semantic_similarity demo.
- Bounded smoke tests (build/encode/batch) + self-diagnosing CI (logs repo me commit).
- Packaging (pyproject.toml), __version__, MIT LICENSE (license -> LICENSE rename),
  CHANGELOG, CITATION.cff, CodeQL, Dependabot, templates.

### Fixed
- from_pretrained error string me raw newlines (SyntaxError) - CI smoke test ne pakda.
- Quickstart CI me import path (PYTHONPATH) issue.
- README wget (HTML page download hota tha) -> gdown commands.
