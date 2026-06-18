from pathlib import Path

# Package-level directory constants
_ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = _ROOT / 'temp'
RENDER_DIR = TEMP_DIR / '_render'
BEAT_EDITOR_DIR = _ROOT / 'beat-editor'

__all__ = ['TEMP_DIR', 'RENDER_DIR', 'BEAT_EDITOR_DIR']