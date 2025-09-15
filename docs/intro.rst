=====================
Welcome to PhysioView
=====================
.. image:: https://badgen.net/badge/python/3.9+/cyan
.. image:: https://badgen.net/badge/license/GPL-3.0/orange
.. image:: https://badgen.net/badge/contributions/welcome/green
.. image:: https://readthedocs.org/projects/heartview/badge/?version=latest
    :target: https://heartview.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

About
-----

PhysioView **(formerly HeartView)** is a Python-based **signal processing and
quality assessment pipeline with an interactive dashboard** designed for wearable
electrocardiograph (ECG), photoplethysmograph (PPG), and electrodermal
activity (EDA) data collected in research settings.

In contrast to other existing tools, PhysioView provides an
open-source graphical user interface intended to increase efficiency and
accessibility for a wider range of researchers who may not otherwise be
able to perform rigorous signal processing and quality checks programmatically.

PhysioView serves as both a diagnostic tool for evaluating data before and
after artifact correction, and as a platform for post-processing
physiological signals. We aim to help researchers make more informed
decisions about data cleaning and processing procedures and the reliabiltiy
of their data when wearable biosensor systems are used.

PhysioView works with data collected from the Actiwave Cardio, Empatica E4,
and other physiological devices outputting data in comma-separated value (CSV)
format.

Features
--------

* **File Reader**: Read and transform raw ECG, PPG, EDA, and accelerometer data
  from Actiwave European Data Format (EDF), archive (ZIP), and CSV files.

* **Configuration File Exporter**: Define and save pipeline parameters in a
  JSON configuration file that can be loaded for use on the same dataset later.

* **Signal Filters**: Filter out noise from baseline wander, muscle (EMG)
  activity, and powerline interference from your physiological signals.

* **Peak Detection**: Extract heartbeats from ECG/PPG and skin conductance
  responses from EDA data.

* **Visualization Dashboard**: View and interact with our signal quality
  assessment charts and signal plots of physiological time series, including
  preprocessed signals and their derived features (e.g., IBI, phasic and tonic
  components).

* **Signal Quality Metrics**: Generate segment-by-segment signal quality
  metrics.

* **Automated Beat Correction**: Appy a beat correction algorithm [|ref1|]
  to automatically correct artifactual beats.

* **Manual Beat Editor**: Manually edit beat locations in cardiac signals.

Citation
--------

If you use this software in your research, please cite the |paper|.

.. code-block:: bibtex

    @inproceedings{Yamane2024,
      author    = {Yamane, N. and Mishra, V. and Goodwin, M.S.},
      title     = {HeartView: An Extensible, Open-Source, Web-Based Signal Quality Assessment Pipeline for Ambulatory Cardiovascular Data},
      booktitle = {Pervasive Computing Technologies for Healthcare. PH 2023},
      series    = {Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering},
      volume    = {572},
      year      = {2024},
      editor    = {Salvi, D. and Van Gorp, P. and Shah, S.A.},
      publisher = {Springer, Cham},
      doi       = {10.1007/978-3-031-59717-6_8},
    }

What's New in PhysioView 1.0
----------------------------

This is the first official release of **PhysioView**, the renamed and
updated continuation of `HeartView <https://heartview.readthedocs.io/en/latest>`_.

Pipeline Enhancements
*********************

- Introduced batch processing support.
- Added postprocessing features, including heart rate variability (HRV)
  extraction.
- Added automated beat correction functionality.
- Introduced EDA processing and signal quality assessment support.

Dashboard Improvements
**********************
- Enabled uploading of ZIP archives for batch processing.
- Added a dropdown menu to select a subject's data from a batch.
- Enabled Beat Editor access within a modal.
- Added rendering of automated and manual beat corrections directly in the
  dashboard's signal plot and SQA charts.
- Introduced export functionality for postprocessed data.

This release consolidates the improvements from **HeartView's final updates** and
marks the beginning of **PhysioView** as its own versioned project. For a
full list of changes, see the full |changelog|.

Installation
------------

The PhysioView source code is available from |GitHub|:

::

   $ git clone https://github.com/cbslneu/physioview.git

See :doc:`installation` for further info.



.. |paper| raw:: html

    <a href="https://doi.org/10.1007/978-3-031-59717-6_8" target="_blank">original paper</a>

.. |ref1| raw:: html

    <a href="https://doi.org/10.3758/s13428-017-0950-2" target="_blank">1</a>

.. |GitHub| raw:: html

    <a href="https://github.com/cbslneu/heartview" target="_blank">GitHub</a>

.. |changelog| raw:: html

    <a href="https://github.com/cbslneu/heartview/blob/main/CHANGELOG.md"
    target="_blank">changelog</a>