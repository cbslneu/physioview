<div align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" 
              srcset="https://raw.githubusercontent.com/cbslneu/heartview/dev/assets/physioview-logo.png"
              height="100">
      <source media="(prefers-color-scheme: light)" 
              srcset="https://raw.githubusercontent.com/cbslneu/heartview/dev/assets/physioview-logo-light.png"
              height="100">
      <img alt="PhysioView Logo"
           src="https://raw.githubusercontent.com/cbslneu/heartview/dev/assets/physioview-logo.png">
    </picture>
    <br>
    <img src="https://badgen.net/badge/python/3.9+/blue">
    <img src="https://badgen.net/badge/license/GPL-3.0/orange">
    <img src="https://badgen.net/badge/contributions/welcome/cyan">
    <a href='https://heartview.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/heartview/badge/?version=latest'>
    </a>
    <br>
    <i>An extensible, open-source, and web-based signal quality assessment pipeline for ambulatory physiological data</i>
    <br>
</div>  
<hr>

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

Currently, PhysioView works with data collected from the Actiwave Cardio, 
Empatica E4, and other devices outputting data in comma-separated 
value (CSV) format.

## Features
* **File Reader**
<br>Read and transform raw ECG, PPG, EDA, and accelerometer data from European Data Format (EDF), archive (ZIP), and CSV files.
* **Configuration File Exporter**
<br>Define and save pipeline parameters in a JSON configuration file that can be loaded for use on the same dataset later.
* **Signal Filters**
<br>Filter out noise from baseline wander, muscle (EMG) activity, and powerline interference from your physiological signals.
* **Peak Detection**
<br>Extract heartbeats from ECG/PPG and skin conductance responses from EDA data.
* **Visualization Dashboard**
<br>View and interact with our signal quality assessment charts and signal plots of physiological time series, including preprocessed signals and their derived features (e.g., IBI, phasic and tonic components).
* **Signal Quality Metrics**
<br>Generate segment-by-segment signal quality metrics.
* **Automated Beat Correction**
<br>Apply a beat correction algorithm [[1](https://doi.org/10.3758/s13428-017-0950-2)] to automatically correct artifactual beats.
* **Manual Beat Editor**
<br>Manually edit beat locations in cardiac signals.

## Citation
If you use this software in your research, please cite [this paper](https://link.springer.com/chapter/10.1007/978-3-031-59717-6_8). :yellow_heart:
```
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
```

## Latest Release [PhysioView 1.0]

This is the first official release of **PhysioView** 🎉  

PhysioView is the renamed and updated continuation of HeartView. Recent 
enhancements made under HeartView are highlighted below, along with new 
features introduced in PhysioView.  

### Pipeline Enhancements:
- Introduced batch processing support.
- Added postprocessing features, including heart rate variability (HRV) 
  extraction.
- Added automated beat correction functionality.
- Introduced EDA processing and signal quality assessment support.

### Dashboard Improvements:
- Enabled uploading of ZIP archives for batch processing.
- Added a dropdown menu to select a subject's data from a batch.
- Enabled Beat Editor access within a modal.
- Added rendering of automated and manual beat corrections directly in the 
  dashboard's signal plot and SQA charts.
- Introduced export functionality for postprocessed data.

### Beat Editor Improvements:
- Simplified the interface by removing static text and file names.

This release consolidates the improvements from **HeartView's final updates** and 
marks the beginning of **PhysioView** as its own versioned project.  

For a full list of changes, see the [full changelog](CHANGELOG.md).

## Installation
1. Clone the PhysioView GitHub repository into a directory of your choice.
```
cd <directory>  # replace <directory> with your directory
```
```
git clone https://github.com/cbslneu/physioview.git
```
2. Set up and activate a virtual environment using Python 3.9 through 3.12 
   inside the `physioview` project directory.  
  ❗️**Note:** If you do not have `virtualenv` installed, run `pip3 install 
virtualenv` before proceeding below.
```
cd physioview
```
```
virtualenv venv -p python3
```
If you are a Mac/Linux user:
```
source venv/bin/activate
```
If you are a Windows user:
```
venv\Scripts\activate
```
3. Install all project dependencies:
```
pip3 install -r requirements.txt
```
### Installation for Beat Editor
**The Beat Editor requires Node (v20.x.x +). Please be sure to install 
Node before proceeding with the installation below.**
<br>
<br> *Run the following code below to check if Node is installed on your machine:*
```
node --version
```
If an error occurs, please refer to this link to install Node on your machine: https://nodejs.org/en/download/package-manager

1. Go to the `beat-editor` directory:
```
cd beat-editor
```
2. Install the required modules for the Beat Editor:
```
npm install
```
3. Go to the `server` folder:
```
cd server
```
4. Install the required modules for the Beat Editor's backend:
```
npm install
```
## PhysioView Dashboard
### Executing
1. Within the activated virtual environment 
(i.e., `source <directory>/physioview/venv/bin/activate`), run the command:
```
python3 app.py
```
2. Open your web browser and go to: http://127.0.0.1:8050/

## PhysioView Beat Editor
### Executing
1. Navigate to the `beat-editor/server` directory and start the backend:
```
cd beat-editor/server
```
```
npm start
```
2. Open another terminal tab or window and navigate back to `beat-editor/`. 
Once there, run `npm start` again to start the front end.

### Terminating
1. Kill the dashboard program: press `CTRL`+`c`.
2. Exit the virtual environment: `deactivate`.
