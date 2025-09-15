.. |br| raw:: html

   <div style="margin-top: 30px"></div>

========
Examples
========

Signal Preprocessing
====================
PhysioView includes built-in functions and methods for preprocessing
ECG, PPG, EDA, and accelerometer signals.

|br|

Electrocardiography (ECG)
-------------------------
When you import the ``physioview`` package, the :class:`ECG.Filters <physioview.pipeline.ECG.Filters>`
and :class:`ECG.BeatDetectors <physioview.pipeline.ECG.BeatDetectors>`
classes are automatically available as ``ECGFilters`` and
``ECGBeatDetectors``, respectively. These provide convenient access to ECG
filtering and beat detection methods.

.. code-block:: python

    import physioview as pv
    import pandas as pd

    # Load the ECG data sampled at 1024 Hz
    ecg = pd.read_csv('sample_ecg_1024hz.csv')
    fs = 1024

    # Filter the ECG signal
    ecg['Filtered'] = pv.ECGFilters(fs).filter_signal(ecg['mV'])

    # Detect beat locations in the filtered ECG signal
    beats_ix = pv.ECGBeatDetectors(fs).manikandan(ecg['Filtered'])

|br|

Photoplethysmography (PPG)
--------------------------
Similarly, the :class:`PPG.Filters <physioview.pipeline.PPG.Filters>`
and :class:`PPG.BeatDetectors <physioview.pipeline.PPG.BeatDetectors>`
classes are automatically available as ``PPGFilters`` and
``PPGBeatDetectors`` when ``physioview`` is imported.

.. code-block:: python

    import physioview as pv
    import pandas as pd

    # Load PPG data sampled at 64 Hz
    ppg = pd.read_csv('sample_ppg_64hz.csv')
    fs = 64

    # Filter the PPG signal
    ppg['Filtered'] = pv.PPGFilters(fs).filter_signal(ppg['BVP'])

    # Detect beat locations in the filtered PPG signal
    beats_ix = pv.PPGBeatDetectors(fs).adaptive_threshold(ppg['Filtered'])

|br|

Electrodermal Activity (EDA)
----------------------------
Access :class:`EDAFilters <physioview.pipeline.EDA.Filters>` as ``EDAFilters``
when importing ``physioview``.

.. code-block:: python

    import physioview as pv
    import pandas as pd

    # Load EDA data sampled at 4 Hz
    eda = pd.read_csv('sample_eda_4hz.csv')
    fs = 4

    # Filter the EDA signal
    eda['Filtered'] = pv.EDAFilters(fs).filter_signal(eda['EDA'])


Feature Extraction
==================

PhysioView also provides feature extraction utilities that convert raw
physiological signals into interpretable measures, such as cardiac interbeat
intervals (IBIs), heart rate variability (HRV) metrics, and EDA components.

Interbeat Intervals & Heart Rate Variability
--------------------------------------------

Extract cardiac features such as IBIs and HRV metrics from an array of beat
locations using :func:`compute_ibis <physioview.physioview.compute_ibis>`
and :func:`compute_hrv <physioview.physioview.compute_hrv>`.

*Note:* PhysioView runs ``flirt`` under the hood. See `FLIRT’s documentation
<https://flirt.readthedocs.io/en/latest/api.html#flirt.hrv.get_hrv_features>`_
for more information about HRV metrics.

.. code-block:: python

    import physioview as pv

    # Compute IBI
    ibi = pv.compute_ibis(ecg, fs, beats_ix, ts_col = 'Timestamp')

    # Compute HRV metrics across 60-sec sliding windows at 15-sec intervals
    hrv = pv.compute_hrv(ecg, fs, beats_ix, window_size = 60,
                         step_size = 15, ts_col = 'Timestamp')

**Outputs:**

The resulting ``ibi`` DataFrame has the same number of rows as the input
``data`` and provides IBI values aligned with detected beat locations in the
signal.

.. code-block:: python

    In [1]: ibi
    Out[2]:
                             Timestamp  IBI
    0       2016-10-14 10:10:51.000000  NaN
    1       2016-10-14 10:10:51.000977  NaN
    2       2016-10-14 10:10:51.001953  NaN
    3       2016-10-14 10:10:51.002930  463.867188
    4       2016-10-14 10:10:51.003906  NaN
    ...                            ...  ...
    381948  2016-10-14 10:17:03.996094  NaN
    381949  2016-10-14 10:17:03.997070  458.984375
    381950  2016-10-14 10:17:03.998047  NaN
    381951  2016-10-14 10:17:03.999023  NaN
    381952  2016-10-14 10:17:04.000000  NaN
    [381953 rows x 2 columns]

The resulting ``hrv`` DataFrame contains HRV features computed over sliding
windows, with each row corresponding to a window and each column to a
specific HRV metric.

.. code-block:: python

    In [3]: hrv.head()
    Out[4]:
                             num_ibis  hrv_mean_nni  ...  hrv_perm_entropy  hrv_svd_entropy
    Timestamp                                    ...
    2016-10-14 10:11:52       129    467.629603  ...          0.995452         0.413629
    2016-10-14 10:12:07       128    468.673706  ...          0.998846         0.385510
    2016-10-14 10:12:22       128    469.207764  ...          0.996257         0.304091
    2016-10-14 10:12:37       128    474.205017  ...          0.996257         0.243242
    2016-10-14 10:12:52       128    469.207764  ...          0.999584         0.120002
    [5 rows x 52 columns]

|br|

Tonic Skin Conductance Level
----------------------------

Compute the tonic skin conductance level (SCL) from an EDA signal across
segments or for the entire signal.

.. code-block:: python

    import physioview as pv

    # Compute tonic SCL across 3-minute windows
    tonic_scl_segment = pv.EDA.compute_tonic_scl(eda['Filtered'], fs, seg_size = 180)

    # Compute tonic SCL for the entire EDA signal
    tonic_scl_entire = pv.EDA.compute_tonic_scl(eda['Filtered'], fs)

|br|

EDA Decomposition
-----------------

Extract the phasic and tonic components of an EDA signal with the convex
optimization approach [1_].

.. code-block:: python

    import physioview as pv

    eda_fs = 4  # sampling rate
    phasic, tonic = pv.EDA.decompose_signal(eda['EDA'], eda_fs)

**Outputs:**

.. code-block:: python

    In [1]: phasic
    Out[2]:
    array([0.        , 0.        , 0.1267762 , ..., 0.15150715, 0.14166146,
           0.13092194])

    In [3]: tonic
    Out[4]:
    array([-3.4407805 , -3.50473019, -3.5572274 , ...,  5.36595805,
            5.36271308,  5.35938202])

.. _1: https://doi.org/10.1109/TBME.2015.2474131

Statistical EDA Features
------------------------

Compute statistical EDA features based on the extracted phasic and tonic
components.

*Note:* PhysioView runs ``flirt`` under the hood. See `FLIRT’s documentation
<https://flirt.readthedocs.io/en/latest/api.html#flirt.eda.get_eda_features>`_
for more information about the outputted EDA features.

.. code-block:: python


.. |Construction| raw:: html

    <img src="_static/under-construction.png" width=700>