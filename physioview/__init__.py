# __init.py
__version__ = '1.0'

# Import signal preprocessing and quality assessment modules and classes
from physioview.pipeline.SQA import Cardio as CardioQA, EDA as EDAQA
from physioview.pipeline.ECG import Filters as ECGFilters, BeatDetectors as ECGBeatDetectors
from physioview.pipeline.PPG import Filters as PPGFilters, BeatDetectors as PPGBeatDetectors
from physioview.pipeline.EDA import Filters as EDAFilters
from physioview.pipeline import ECG, PPG, EDA, SQA

# Import public symbols from physioview.py
from physioview.physioview import *
import physioview.physioview as _physioview

# Create alias for `EDA.compute_features()`
def compute_eda_features(signal, fs, window_size, step_size):
    return EDA.compute_features(signal, fs, window_size, step_size)

# Expose classes and methods
__all__ = [
    'CardioQA',
    'EDAQA',
    'ECGFilters',
    'ECGBeatDetectors',
    'PPGFilters',
    'PPGBeatDetectors',
    'EDAFilters',
    'ECG',
    'PPG',
    'EDA',
    'SQA',
    'compute_eda_features',
]

# Extend __all__ with everything public from physioview.py
__all__ += _physioview.__all__