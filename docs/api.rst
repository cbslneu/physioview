=============
API Reference
=============

Data Preprocessing
==================

ECG
...

.. automodule:: physioview.pipeline.ECG
   :members:

PPG
...

.. automodule:: physioview.pipeline.PPG
   :members:

EDA
...

.. automodule:: physioview.pipeline.EDA
   :members:

ACC
...

.. automodule:: physioview.pipeline.ACC
   :members:

Signal Quality Assessment
=========================

.. automodule:: physioview.pipeline.SQA
   :members:

Device-Specific Methods
=======================
Actiwave Cardio
...............

.. autoclass:: physioview.physioview.Actiwave
    :members:
    :undoc-members:

Empatica E4
...........

.. autoclass:: physioview.physioview.Empatica
    :members:
    :undoc-members:

Other Signal Processing Functions
=================================
.. _compute_ibis:

.. autofunction:: physioview.physioview.compute_ibis

.. _compute_hrv:

.. autofunction:: physioview.physioview.compute_hrv

.. _plot_signal:

.. autofunction:: physioview.physioview.plot_signal

Beat Editor Functions
=====================
.. _write-beat-editor-file:

.. autofunction:: physioview.physioview.write_beat_editor_file

.. _process_beat_edits:

.. autofunction:: physioview.physioview.process_beat_edits