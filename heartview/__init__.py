# __init.py
__version__ = '2.1.2'

# Import convenient modules and classes
from heartview import heartview
from heartview.pipeline.SQA import Cardio

# Initialize SQA classes
def cardio_sqa(fs):
    """Initialize the heartview.pipeline.SQA.Cardio class containing signal
    quality assessment functions for ECG and PPG data."""
    return Cardio(fs)

# Expose functions
__all__ = ['cardio_sqa']