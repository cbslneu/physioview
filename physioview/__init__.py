# __init.py
__version__ = '1.0'

# Import convenient modules and classes
from physioview import physioview
from physioview.pipeline.SQA import Cardio, EDA
from physioview.pipeline.ECG import Filters as ecg_filters
from physioview.pipeline.ECG import BeatDetectors as ecg_beat_detectors
from physioview.pipeline.PPG import Filters as ppg_filters
from physioview.pipeline.PPG import BeatDetectors as ppg_beat_detectors

# Initialize SQA classes
def cardio_sqa(fs: int) -> 'Cardio':
    """Initialize the physioview.pipeline.SQA.Cardio class containing signal
    quality assessment functions for ECG and PPG data."""
    return Cardio(fs)

def eda_sqa(
    fs,
    eda_min: float = 0.2,
    eda_max: float = 40,
    eda_max_slope: float = 5,
    temp_min: float = 20,
    temp_max: float = 40,
    invalid_spread_dur: float = 2.5
) -> 'EDA':
    """Initialize the physioview.pipeline.SQA.EDA class containing signal
    quality assessment functions for EDA data.

    Parameters
    ----------
    fs : int
        The sampling rate of the EDA recording.
    eda_min : float, optional
        The minimum acceptable value for EDA data in microsiemens; by
        default, 0.2 uS.
    eda_max : float, optional
        The maximum acceptable value for EDA data in microsiemens; by
        default, 40 uS.
    eda_max_slope : float, optional
        The maximum slope of EDA data in microsiemens per second; by
        default, 5 uS/sec.
    temp_min : float, optional
        The minimum acceptable temperature in degrees Celsius; by
        default, 20.
    temp_max : float, optional
        The maximum acceptable temperature in degrees Celsius; by
        default, 40.
    invalid_spread_dur : float, optional
        The transition radius for artifacts in seconds; by default,
        2.5 seconds.
"""
    return EDA(fs)

# Expose functions
__all__ = ['cardio_sqa', 'ecg_filters', 'ecg_beat_detectors',
           'ppg_filters', 'ppg_beat_detectors']