"""
Beat Editor and automated beat correction utilities for the Physioview
Dashboard.

This module provides functionality for both manual beat editing via the
external Beat Editor application and automated beat correction procedure.

All functions in this module are intended for internal use by the dashboard
and should not be called directly from external code.
"""

from typing import Tuple
from os import path
from physioview.pipeline import SQA
from physioview.physioview import compute_ibis
from requests import get as http_get
import numpy as np
import pandas as pd

# =============================== BEAT EDITOR ================================
def _check_beat_editor_status() -> bool:
    """Check whether the Beat Editor app is running."""
    try:
        response = http_get('http://localhost:3000', timeout = 5)
        return response.status_code == 200
    except:
        return False

def _create_beat_editor_file(
    data: pd.DataFrame,
    filename: str
) -> None:
    """Create a Beat Editor JSON file."""
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    for col in ['PPG', 'BVP', 'ECG']:
        if col in data.columns:
            data.rename(columns = {col: 'Signal'}, inplace = True)
            break
    if 'Filtered' in data.columns:
        data = data.drop(columns = ['Signal'])
    root_dir = '/'.join(path.dirname(path.abspath(__file__)).split('/')[:-2])
    target_dir = path.join(root_dir, 'beat-editor', 'data')
    file_path = path.join(target_dir, f"{filename}_edit.json")
    data.to_json(file_path, orient = 'records', lines = False)

def _map_beat_edits(
    edited_ix: np.ndarray,
    beat_editor_fs: int,
    target_fs: int,
) -> np.ndarray:
    """Map edited beat indices to another time grid."""
    scale = target_fs / beat_editor_fs
    mapped_edits_ix = np.rint(edited_ix * scale).astype(int)
    return mapped_edits_ix

# ======================== AUTOMATED BEAT CORRECTION =========================
def _correct_beats(
    signal: pd.DataFrame,
    fs: int,
    beats_ix: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Correct the beats in a signal."""
    signal = signal.copy()
    sqa = SQA.Cardio(fs)
    beats_ix_corrected, _, _, _, = sqa.correct_interval(
        beats_ix, print_estimated_hr = False)
    signal.loc[beats_ix_corrected, 'Corrected'] = 1
    ts_col = 'Timestamp' if 'Timestamp' in signal.columns else None
    ibi_corrected = compute_ibis(signal, fs, beats_ix_corrected, ts_col)
    return signal, beats_ix_corrected, ibi_corrected

def _accept_beat_corrections(
    signal: pd.DataFrame,
    fs: int,
    artifact_method: str,
    artifact_tol: float
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Accept the suggested automatic beat corrections in a signal."""
    signal = signal.copy()

    # Save original beat indices
    signal.loc[signal['Beat'] == 1, 'Original Beat'] = 1

    # Reset beat column
    signal['Beat'] = None

    # Update beat column with corrected beats
    signal.loc[signal['Corrected'] == 1, 'Beat'] = 1
    signal.drop(columns = ['Corrected'], inplace = True)

    # Update artifacts
    beats_ix = signal.loc[signal['Beat'] == 1].index.values
    sqa = SQA.Cardio(fs)
    artifacts_ix = sqa.identify_artifacts(
        beats_ix, method = artifact_method, tol = artifact_tol,
        initial_hr = 'auto')
    signal['Artifact'] = None
    signal.loc[artifacts_ix, 'Artifact'] = 1
    return signal, beats_ix, artifacts_ix

def _revert_beat_corrections(
    signal: pd.DataFrame,
    fs: int,
    artifact_method: str,
    artifact_tol: float
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Revert the beat corrections in a signal."""
    signal = signal.copy()
    signal['Beat'] = None
    signal.loc[signal['Original Beat'] == 1, 'Beat'] = 1
    beats_ix = signal.loc[signal['Beat'] == 1].index.values
    sqa = SQA.Cardio(fs)
    artifacts_ix = sqa.identify_artifacts(
        beats_ix, method = artifact_method, tol = artifact_tol,
        initial_hr = 'auto')
    signal['Artifact'] = None
    signal.loc[artifacts_ix, 'Artifact'] = 1
    return signal, beats_ix, artifacts_ix