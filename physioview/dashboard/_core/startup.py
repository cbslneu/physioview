"""
Startup utilities for the PhysioView Dashboard.

This module contains functions for managing temporary directories and files
used by the dashboard during operation. These functions handle directory
creation at startup and cleanup of temporary data between sessions.

All functions in this module are intended for internal use by the dashboard
and should not be called directly from external code.
"""

from pathlib import Path
from shutil import rmtree

def _make_subdirs() -> None:
    """Create necessary subdirectories for dashboard operation (temp files,
    beat editor data)."""
    (Path('.') / 'temp').mkdir(
        parents = True, exist_ok = True)
    (Path('.') / 'beat-editor' / 'data').mkdir(
        parents = True, exist_ok = True)

def _clear_temp() -> None:
    """Remove all files and subdirectories from the temp directory."""
    temp = Path('.') / 'temp'
    if not temp.exists():
        return
    for item in temp.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            rmtree(item)

def _clear_edits() -> None:
    """Clear beat editor data and saved edits."""
    beat_editor_paths = [
        Path('.') / 'beat-editor' / 'data',
        Path('.') / 'beat-editor' / 'saved'
    ]
    for p in beat_editor_paths:
        if not p.exists():
            continue
        for item in p.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                rmtree(item)