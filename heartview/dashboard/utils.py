import zipfile
from typing import Callable, Optional, Union
from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile
from scipy.signal import filtfilt, firwin
from heartview.pipeline import ECG, SQA, PPG
from heartview.heartview import compute_ibis
from dash import html
from requests import get as http_get
from os import path
from io import BytesIO
from time import sleep
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pyedflib
import json

# ============================= Startup Functions ============================
def _make_subdirs():
    (Path('.') / 'temp').mkdir(
        parents = True, exist_ok = True)
    (Path('.') / 'beat-editor' / 'data').mkdir(
        parents = True, exist_ok = True)
    (Path('.') / 'beat-editor' / 'export').mkdir(
        parents = True, exist_ok = True)

def _clear_temp():
    temp = Path('.') / 'temp'
    if not temp.exists():
        return
    for item in temp.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            rmtree(item)

def _clear_edits():
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

# ======================= HearView Pipeline Functions ========================
def _preprocess_cardiac(
    data: pd.DataFrame,
    dtype: str,
    fs: int,
    seg_size: int,
    beat_detector: str,
    artifact_method: str,
    artifact_tol: float,
    filt_on: bool,
    acc_data: Optional[pd.DataFrame] = None,
    downsample: bool = True,
    set_progress: Callable[[tuple[Union[int, float], str]], None] = None
) -> tuple:
    """Run the HeartView pipeline on ECG/PPG data."""
    is_preprocessed = False if not filt_on else True
    if dtype == 'ECG':
        filt = ECG.Filters(fs)
        detect_beats = ECG.BeatDetectors(fs, is_preprocessed)
    elif dtype in ('PPG', 'BVP'):
        filt = PPG.Filters(fs)
        detect_beats = PPG.BeatDetectors(fs, is_preprocessed)

    preprocessed_data = data.copy()

    # Filter ECG and detect beats
    if set_progress is not None:
        total_progress = 6                              # 33% progress
        perc = (2 / total_progress) * 100
        set_progress((perc * 100, f'{perc:.0f}'))
    if filt_on:
        preprocessed_data['Filtered'] = filt.filter_signal(
            preprocessed_data[dtype])
        if set_progress is not None:
            perc = (3 / total_progress) * 100           # 50% progress
            set_progress((perc * 100, f'{perc:.0f}%'))
        beats_ix = getattr(detect_beats, beat_detector)(
            preprocessed_data['Filtered'])
    else:
        beats_ix = getattr(detect_beats, beat_detector)(
            preprocessed_data[dtype])
    preprocessed_data.loc[beats_ix, 'Beat'] = 1
    preprocessed_data.insert(
        0, 'Segment', preprocessed_data.index // (seg_size * fs) + 1)
    if set_progress is not None:
        perc = (3.5 / total_progress) * 100             # 58% progress
        set_progress((perc * 100, f'{perc:.0f}%'))

    # Identify artifactual beats
    sqa = SQA.Cardio(fs)
    artifacts_ix = sqa.identify_artifacts(
        beats_ix, method = artifact_method, tol = artifact_tol)
    preprocessed_data.loc[artifacts_ix, 'Artifact'] = 1
    if set_progress is not None:
        perc = (4 / total_progress) * 100               # 66% progress
        set_progress((perc * 100, f'{perc:.0f}%'))

    # Compute IBIs and SQA metrics
    ts_col = 'Timestamp' if 'Timestamp' in preprocessed_data.columns else None
    ibi = compute_ibis(preprocessed_data, fs, beats_ix, ts_col)
    metrics = sqa.compute_metrics(
        preprocessed_data, beats_ix, artifacts_ix, seg_size = seg_size,
        show_progress = False)
    if set_progress is not None:
        perc = (5 / total_progress) * 100               # 83% progress
        set_progress((perc * 100, f'{perc:.0f}%'))

    # Downsample data to at least 125 Hz for quicker plot rendering
    if downsample:
        ds_data, ds_ibi, ds_acc, ds_fs = _downsample_data(
            preprocessed_data, fs, 'ECG', beats_ix, artifacts_ix,
            acc = acc_data)
        return preprocessed_data, ibi, metrics, ds_data, ds_ibi, ds_acc, ds_fs
    else:
        return preprocessed_data, ibi, metrics

# ===================== HeartView Dashboard UI Functions =====================
def _check_csv(name) -> bool:
    """Check if a CSV file is valid."""
    return (
        name.endswith('.csv')
        and not name.startswith(('__MACOSX/', '.'))
        and not path.basename(name).startswith('.')
        and not name.endswith('/')
    )

def _check_edf(edf) -> str:
    """Check whether the EDF uploaded is a valid Actiwave Cardio file."""
    f = pyedflib.EdfReader(edf)
    signals = f.getSignalLabels()
    if any('ECG0' in s for s in signals):
        return 'ECG'
    else:
        return 'invalid'

def _get_configs() -> list[str]:
    cfg_dir = Path('.') / 'configs'
    cfgs = [f.name for f in cfg_dir.iterdir() if f.is_file() and
            not f.name.startswith('.')]
    if len(cfgs) > 0:
        return cfgs
    else:
        return []

def _create_configs(
    source: str,
    dtype: str,
    fs: int,
    seg_size: int,
    artifact_method: str,
    artifact_tol: float,
    filter_on: bool,
    headers: Optional[dict] = None
) -> str:
    """Create a JSON-formatted configuration file of user SQA parameters."""

    # Save user configuration
    configs = {'source': source,
               'data type': dtype,
               'sampling rate': fs,
               'segment size': seg_size,
               'filters': filter_on,
               'artifact identification method': artifact_method,
               'artifact tolerance': artifact_tol}

    if headers is not None:
        configs['headers'] = headers

    # Serialize JSON
    json_object = json.dumps(configs)

    return json_object

def _load_config(filename: str) -> dict:
    """Load a JSON configuration file into a dictionary."""
    cfg = open(filename)
    configs = json.load(cfg)
    return configs

def _export_sqa(
    file: str,
    data_type: str,
    type: str
) -> None:
    """Export the SQA summary data in Zip or Excel format."""
    temp_dir = Path('temp')
    downloads_dir = Path('downloads')
    downloads_dir.mkdir(parents = True, exist_ok = True)

    files = [temp_dir / f'{file}_SQA.csv']

    if data_type == 'E4':
        files += [
            temp_dir / f'{file}_BVP.csv',
            temp_dir / f'{file}_ACC.csv',
            temp_dir / f'{file}_IBI.csv',
            temp_dir / f'{file}_EDA.csv'
        ]
    elif data_type == 'Actiwave':
        files += [
            temp_dir / f'{file}_ECG.csv',
            temp_dir / f'{file}_ACC.csv',
            temp_dir / f'{file}_IBI.csv'
        ]
    else:  # Generic PPG or CSV input
        files += [
            temp_dir / f'{file}_ECG.csv',
            temp_dir / f'{file}_IBI.csv'
        ]
        acc_file = temp_dir / f'{file}_ACC.csv'
        if acc_file.exists():
            files.append(acc_file)

    if type == 'zip':
        zip_path = downloads_dir / f'{file}_sqa_summary.zip'
        with ZipFile(zip_path, 'w') as archive:
            for csv in files:
                archive.write(csv)
    elif type == 'excel':
        excel_path = downloads_dir / f'{file}_sqa_summary.xlsx'
        with pd.ExcelWriter(excel_path) as xlsx:
            for csv in files:
                df = pd.read_csv(csv)
                sheet_name = csv.stem
                df.to_excel(xlsx, sheet_name = sheet_name, index = False)
    return None

def _get_csv_headers(
    csv: str
) -> list[str]:
    """Get the headers of a user-uploaded CSV file in a list."""
    initial = pd.read_csv(csv, nrows = 1)
    headers = initial.columns.tolist()
    return headers

def _setup_data_ts(
    csv: str,
    dtype: str,
    dropdown: list[str]
) -> pd.DataFrame:
    """Read uploaded CSV data into a data frame with timestamps."""
    df = pd.read_csv(csv, usecols = dropdown)
    if len(dropdown) > 2:
        df = df.rename(columns = {
            dropdown[0]: 'Timestamp',
            dropdown[1]: dtype,
            dropdown[2]: 'X',
            dropdown[3]: 'Y',
            dropdown[4]: 'Z'})
    else:
        df = df.rename(columns = {
            dropdown[0]: 'Timestamp',
            dropdown[1]: dtype})
    return df

def _setup_data_samples(
    csv: str, dtype:
    str, dropdown:
    list[str]
) -> pd.DataFrame:
    """Read uploaded CSV data into a data frame with sample counts."""
    df = pd.read_csv(csv, usecols = dropdown)
    if len(dropdown) > 1:
        df = df.rename(columns = {
            dropdown[0]: dtype,
            dropdown[1]: 'X',
            dropdown[2]: 'Y',
            dropdown[3]: 'Z'})
    else:
        df = df.rename(columns = {
            dropdown[0]: dtype})
    samples = np.arange(len(df)) + 1
    df.insert(0, 'Sample', samples)
    return df

def _downsample_data(
    df: pd.DataFrame,
    fs: int,
    signal_type: str,
    beats_ix: Union[list[int], np.ndarray],
    artifacts_ix: Union[list[int], np.ndarray],
    ds_target: int = 250,
    acc: Optional[pd.DataFrame] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Downsample pre-processed cardio data and any acceleration data for
    quicker plot rendering on the dashboard."""

    # Choose x and y columns
    x_col = 'Timestamp' if 'Timestamp' in df.columns else 'Sample'
    y_col = 'Filtered' if 'Filtered' in df.columns else signal_type

    ds_factor = max(1, int(fs) // ds_target)
    ds_fs = int(fs / ds_factor)
    ds_idx = np.arange(0, len(df), ds_factor)

    # Zero-phase anti-alias before decimation
    b = firwin(numtaps = 129, cutoff = 0.45 / ds_factor)
    y_src = df[y_col].to_numpy()
    y_aa = filtfilt(b, [1.0], y_src)
    y_dec = y_aa[::ds_factor]
    ds = pd.DataFrame({
        x_col: df[x_col].iloc[ds_idx].to_numpy(),
        y_col: y_dec
    })


    # Rescale detected and artifactual beat indices
    down_beats = np.rint(
        beats_ix / ds_factor).astype(int).clip(0, len(ds) - 1)
    down_artifacts = np.rint(
        artifacts_ix / ds_factor).astype(int).clip(0, len(ds) - 1)
    ds.loc[down_beats, 'Beat'] = 1
    ds.loc[down_artifacts, 'Artifact'] = 1

    # Downsample IBI data
    ds_ibi = compute_ibis(ds, ds_fs, down_beats, ts_col = x_col)

    # Downsample acceleration data
    if acc is not None:
        acc_mag = acc['Magnitude'].to_numpy()

        # Anti-alias and then decimate ACC signal
        b = firwin(numtaps = 129, cutoff = 0.45 / ds_factor)
        acc_aa = filtfilt(b, [1.0], acc_mag)
        acc_dec = acc_aa[::ds_factor]
        ds_acc = pd.DataFrame({x_col: df[x_col].iloc[ds_idx].to_numpy(),
                               'Magnitude': acc_dec})
    else:
        ds_acc = None
    return ds, ds_ibi, ds_acc, ds_fs

def _cardiac_summary_table(sqa_df: pd.DataFrame) -> dbc.Table:
    """Display the SQA summary table."""

    # Calculate average heart rate
    valid_df = sqa_df[sqa_df.Invalid != 1].copy().reset_index(drop = True)
    valid_ix = np.where(np.diff(valid_df['N Detected']) < 10)[0]
    valid_df = valid_df.loc[valid_ix].reset_index(drop = True)
    avg_hr = '{0:.2f}'.format(valid_df['N Detected'].mean())
    missing_n = len(sqa_df.loc[sqa_df['N Missing'] > 0])
    artifact_n = len(sqa_df.loc[sqa_df['N Artifact'] > 0])
    invalid_n = len(sqa_df.loc[sqa_df['Invalid'] == 1])
    invalid_prop = '{0:.2f}%'.format(
        (invalid_n / sqa_df['Segment'].max()) * 100)
    avg_missing = '{0:.2f}%'.format(sqa_df['% Missing'].mean())
    avg_artifact = '{0:.2f}%'.format(
        sqa_df.loc[sqa_df['% Artifact'] > 0, '% Artifact'].mean())

    data = [
        ('Average Heart Rate', avg_hr),
        ('Segments with Missing Beats', missing_n),
        ('Segments with Artifactual Beats', artifact_n),
        ('Segments with Invalid Beats', invalid_n),
        ('% Invalid Data', invalid_prop),
        ('Average % Missing Beats/Segment', avg_missing),
        ('Average % Artifactual Beats/Segment', avg_artifact)
    ]

    # Manually build the table body
    rows = [
        html.Tr([
            html.Td(label),
            html.Td(value)
        ]) for label, value in data
    ]

    # Wrap in a dbc.Table
    table = dbc.Table(
        rows,
        className = 'segmentTable',
        striped = False,
        bordered = False,
        hover = False
    )

    return table, data

def _make_excel(
    files: list[Path],
    max_rows: int = 1_000_000,
    set_progress: Callable[[tuple[Union[int, float], str]], None] = None,
    progress_start: int = 0,
    progress_total: Optional[int] = None,
) -> BytesIO:
    """Create an Excel workbook from a list of files with optional
    per-file incrementation of a progress bar."""
    if set_progress is not None:
        n_files = len(files)
        total_progress = progress_total if progress_total is not None \
            else n_files + progress_start

    out = BytesIO()
    with pd.ExcelWriter(out) as xlsx:
        for i, file_path in enumerate(files):

            # Write quality summary text file separately
            if str(file_path).endswith('.txt'):
                with open(str(file_path), 'r') as txt_file:
                    lines = txt_file.readlines()
                summary_data = [line.strip().split(':', 1) for line in lines
                                if ':' in line]
                summary_df = pd.DataFrame(
                    summary_data, columns = ['Metric', 'Value'])
                summary_df.to_excel(
                    xlsx, sheet_name = 'Quality Summary', index = False)

            # Write all other CSV files
            else:
                df = pd.read_csv(file_path)
                if 'cleaned' in str(file_path):
                    fname = file_path.stem.split('_')[-2]
                else:
                    fname = file_path.stem.split('_')[-1]
                num_sheets = (len(df) + max_rows - 1) // max_rows
                for j in range(num_sheets):
                    start_row = j * max_rows
                    end_row = min((j + 1) * max_rows, len(df))
                    df_chunk = df.iloc[start_row:end_row]
                    if df_chunk.empty:
                        continue  # prevent writing past max row
                    sheet_name = f'{fname}_{j + 1}' if num_sheets > 1 else fname
                    sheet_name = sheet_name[:31]  # sheet name limit
                    df_chunk.to_excel(
                        xlsx, sheet_name = sheet_name, index = False)

            # Update progress bar
            if set_progress is not None:
                remaining = max(total_progress - progress_start, 0)
                frac = (i + 1) / n_files
                progress = (progress_start + remaining
                            * frac) / total_progress * 100
                set_progress((progress, f'{progress:.0f}%'))
                sleep(0.3)

    out.seek(0)
    return out

def _make_zip(
    files: list[Path],
    set_progress: Callable[[tuple[Union[int, float], str]], None] = None,
    progress_start: int = 0,
    progress_total: Optional[int] = None
) -> BytesIO:
    """Build a Zip archive file from a list of files with optional
    per-file incrementation of a progress bar."""
    if set_progress is not None:
        n_files = len(files)
        total_progress = progress_total if progress_total is not None \
            else n_files + progress_start

    out = BytesIO()
    with ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, file_path in enumerate(files):
            file_name = file_path.name
            with open(file_path, 'rb') as f:
                zf.writestr(file_name, f.read())

                # If set_progress, update progress
                remaining = max(total_progress - progress_start, 0)
                frac = (i + 1) / n_files
                progress = (progress_start + remaining
                            * frac) / total_progress * 100
                set_progress((progress, f'{progress:.0f}%'))
                sleep(0.5)
    out.seek(0)
    return out

def _blank_fig(context) -> go.Figure:
    """Display the default blank figure."""
    fig = go.Figure(go.Scatter(x = [], y = []))
    fig.update_layout(template = None,
                      paper_bgcolor = 'rgba(0, 0, 0, 0)',
                      plot_bgcolor = 'rgba(0, 0, 0, 0)')
    fig.update_xaxes(showgrid = False,
                     showticklabels = False,
                     zeroline = False)
    fig.update_yaxes(showgrid = False,
                     showticklabels = False,
                     zeroline = False)
    if context == 'pending':
        fig.add_annotation(text = '<i>Input participant data to view...</i>',
                           xref = 'paper', yref = 'paper',
                           font = dict(family = 'Poppins',
                                       size = 14,
                                       color = '#3a4952'),
                           x = 0.5, y = 0.5, showarrow = False)
    if context == 'none':
        fig.add_annotation(text = '<i>No data to view.</i>',
                           xref = 'paper', yref = 'paper',
                           font = dict(family = 'Poppins',
                                       size = 14,
                                       color = '#3a4952'),
                           x = 0.5, y = 0.5, showarrow = False)
    return fig

def _blank_table() -> dbc.Table:
    """Display the default blank table."""
    summary = pd.DataFrame({
        'Metric': [
            'Average Heart Rate',
            'Segments with Missing Beats',
            'Segments with Artifactual Beats',
            'Segments with Invalid Beats',
            '% Invalid Data',
            'Average % Missing Beats/Segment',
            'Average % Artifactual Beats/Segment'
        ],
        'Value': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    })

    # Generate table with headers
    table = dbc.Table.from_dataframe(
        summary,
        index = False,
        className = 'segmentTable',
        striped = False,
        hover = False,
        bordered = False
    )
    # Remove the header row (Thead)
    table.children = [child for child in table.children if not isinstance(child, html.Thead)]
    return table

def _check_beat_editor_status() -> bool:
    """Check whether the beat editor app is running."""
    try:
        response = http_get('http://localhost:3000', timeout = 5)
        return response.status_code == 200
    except:
        return False

def _create_beat_editor_file(
    data: pd.DataFrame,
    filename: str
) -> None:
    """Create a beat editor JSON file."""
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