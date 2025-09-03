import zipfile
from typing import Callable, Optional, Union
from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile
from scipy.signal import filtfilt, firwin
from heartview.pipeline import ECG, EDA, PPG, SQA
from heartview.heartview import compute_ibis
from dash import html
from requests import get as http_get
from os import path
from io import BytesIO, StringIO
from time import sleep
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pyedflib
import json
import base64

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
    downsample: bool = True
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
    if filt_on:
        preprocessed_data['Filtered'] = filt.filter_signal(
            preprocessed_data[dtype])
        beats_ix = getattr(detect_beats, beat_detector)(
            preprocessed_data['Filtered'])
    else:
        beats_ix = getattr(detect_beats, beat_detector)(
            preprocessed_data[dtype])
    preprocessed_data.loc[beats_ix, 'Beat'] = 1
    preprocessed_data.insert(
        0, 'Segment', preprocessed_data.index // (seg_size * fs) + 1)

    # Identify artifactual beats
    sqa = SQA.Cardio(fs)
    artifacts_ix = sqa.identify_artifacts(
        beats_ix, method = artifact_method, tol = artifact_tol)
    preprocessed_data.loc[artifacts_ix, 'Artifact'] = 1

    # Compute IBIs and SQA metrics
    ts_col = 'Timestamp' if 'Timestamp' in preprocessed_data.columns else None
    if ts_col == 'Timestamp':
        unix_fmt = _check_unix(preprocessed_data.Timestamp)
        if unix_fmt is not None:
            preprocessed_data.Timestamp = pd.to_datetime(
                preprocessed_data.Timestamp, unit = unix_fmt)
    ibi = compute_ibis(preprocessed_data, fs, beats_ix, ts_col)
    metrics = sqa.compute_metrics(
        preprocessed_data, beats_ix, artifacts_ix, seg_size = seg_size,
        show_progress = False)

    # Downsample data to at least 250 Hz for quicker plot rendering
    if downsample:
        ds_data, ds_ibi, _, ds_acc, ds_fs = _downsample_data(
            preprocessed_data, fs, 'ECG', beats_ix, artifacts_ix,
            acc = acc_data)
        return preprocessed_data, ibi, metrics, ds_data, ds_ibi, ds_acc, ds_fs
    else:
        return preprocessed_data, ibi, metrics

def _correct_beats(
    signal: pd.DataFrame,
    fs: int,
    beats_ix: np.ndarray,
):
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
):
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
):
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

def _preprocess_eda(
    data: pd.DataFrame,
    fs: int,
    rs: Optional[int] = None,
    temp: Optional[np.ndarray] = None,
    seg_size: int = 60,
    filt_on: bool = True,
    scr_on: bool = True,
    scr_amp: float = 0.1,
    eda_min: float = 0.2,
    eda_max: float = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the PhysioView data processing pipeline on EDA data."""
    preprocessed_data = data.copy()
    signal = preprocessed_data['EDA'].values
    has_ts = 'Timestamp' in preprocessed_data.columns
    ts_col = 'Timestamp' if has_ts else 'Sample'

    # Decide target sampling rate
    target_fs = rs if rs is not None else (8 if fs > 8 else fs)

    # Resample data if requested
    if target_fs != fs:
        signal_rs = EDA.resample(signal, fs, target_fs)
        fs_eff = target_fs
    else:
        signal_rs = signal
        fs_eff = fs

    # Build timestamps/sample indices
    n_samples = len(signal_rs)
    if has_ts:
        unix_fmt = _check_unix(preprocessed_data.Timestamp)
        if unix_fmt is not None:
            preprocessed_data.Timestamp = pd.to_datetime(
                preprocessed_data.Timestamp, unit = unix_fmt)
        t0 = preprocessed_data['Timestamp'].iloc[0]
        step = pd.to_timedelta(1, unit = 's') / fs_eff
        ts_rs = pd.date_range(start = t0, periods = n_samples, freq = step)
    else:
        ts_rs = np.arange(n_samples)

    # Create resampled data
    data_rs = pd.DataFrame({ts_col: ts_rs, 'EDA': signal_rs})
    preprocessed_data = data_rs
    fs = fs_eff  # update sampling rate to resampled rate

    # Apply filter if requested
    if filt_on:
        filter_eda = EDA.Filters(fs)
        preprocessed_data['Filtered'] = filter_eda.filter_signal(
            preprocessed_data['EDA'], fs)
        is_preprocessed = True
        if temp is not None:
            preprocessed_data['TEMP'] = filter_eda.moving_average(
                temp, window_len = 5)
            temp = preprocessed_data['TEMP'].values
    else:
        is_preprocessed = False

    # Decompose to phasic and tonic components with convex optimization
    phasic, tonic = EDA.decompose_eda(
        preprocessed_data['EDA'], fs, show_progress = False)
    preprocessed_data['Phasic'] = phasic
    preprocessed_data['Tonic'] = tonic

    # Detect SCR peaks if requested
    if scr_on:
        scr_ix = EDA.detect_scr_peaks(preprocessed_data['Phasic'],
                                      min_amp_thresh = scr_amp)
        preprocessed_data.loc[scr_ix, 'SCR'] = 1

    # Compute quality metrics
    edaqa = SQA.EDA(fs, eda_min, eda_max)
    ycol = 'Filtered' if 'Filtered' in preprocessed_data.columns else 'EDA'
    eda_validity = edaqa.get_validity_metrics(preprocessed_data[ycol])
    preprocessed_data = pd.concat([
        preprocessed_data, eda_validity['Invalid']], axis = 1)
    eda_quality = edaqa.get_quality_metrics(preprocessed_data[ycol])
    preprocessed_data = pd.concat([
        preprocessed_data, eda_quality[eda_quality.columns[-3:]]], axis = 1)
    metrics = edaqa.compute_metrics(
        preprocessed_data[ycol], temp, is_preprocessed, seg_size,
        show_progress = False)

    return preprocessed_data, metrics

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


def _check_unix(ts: pd.Series) -> Union[str, None]:
    """Check whether a given timestamps column contains Unix timestamps in
    s, ms, or µs."""
    try:
        vals = pd.to_numeric(ts, errors = "coerce").dropna()
    except Exception:
        return None
    if vals.empty:
        return None
    median_val = vals.median()
    if 1e8 < median_val < 2e9:
        return 's'
    elif 1e11 < median_val < 2e13:
        return 'ms'
    elif 1e14 < median_val < 2e16:
        return 'us'
    else:
        return None

def _create_configs(
    source: str,
    dtype: str,
    fs: int,
    seg_size: int,
    artifact_method: str,
    artifact_tol: float,
    filter_on: bool,
    scr_on: bool,
    scr_amp: float,
    headers: Optional[dict] = None
) -> str:
    """Create a JSON-formatted configuration file of user SQA parameters."""

    # Save user configuration
    configs = {'source': source,
               'data type': dtype,
               'sampling rate': fs,
               'segment size': seg_size,
               'filters': filter_on,
               'scr detection': scr_on,
               'scr amplitude': scr_amp,
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

def _parse_temp_csv(contents: str) -> pd.DataFrame:
    """Parse temperature data uploaded with dcc.Upload component."""
    content_type, content_string = contents.split(',')
    raw = base64.b64decode(content_string)
    buf = StringIO(raw.decode('utf-8'))
    return pd.read_csv(buf)

def _setup_data(
    csv: str,
    dtype: str,
    dropdowns: list[str],
    temp_var: Optional[str] = None,
    has_ts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and map columns of uploaded CSV data to variables."""
    cols = dropdowns.copy()

    # Check if acceleration data is provided
    has_acc = len(dropdowns) > (1 if has_ts else 0) + 1

    # Add temperature column if given
    has_temp = temp_var is not None
    if has_temp: cols.append(temp_var)

    # Read data with the given columns
    df = pd.read_csv(csv, usecols = cols)
    df = df[cols].copy()

    # Rename columns
    rename_map, i = {}, 0
    if has_ts:
        rename_map[dropdowns[i]] = 'Timestamp'
        i += 1
    rename_map[dropdowns[i]] = dtype
    i += 1
    if has_acc:
        for ax in ['X', 'Y', 'Z']:
            rename_map[dropdowns[i]] = ax
            i += 1
    if has_temp:
        rename_map[temp_var] = 'TEMP'
    df.rename(columns = rename_map, inplace = True)

    # Insert 'Sample column' if no timestamp
    if not has_ts:
        df.insert(0, 'Sample', np.arange(len(df)) + 1)
        ts_col = 'Sample'
    else:
        ts_col = 'Timestamp'

    # Build signal DataFrame with 'TEMP' if it exists
    data_cols = [ts_col, dtype]
    if 'TEMP' in df.columns:
        data_cols.append('TEMP')
    data = df[data_cols]

    # Build acceleration DataFrame
    acc = None
    if has_acc:
        acc_cols = [ts_col, 'X', 'Y', 'Z']
        acc = df[acc_cols]

    return data, acc

def _downsample_data(
    df: pd.DataFrame,
    fs: int,
    signal_type: str,
    beats_ix: Union[list[int], np.ndarray],
    artifacts_ix: Union[list[int], np.ndarray],
    corrected_beats_ix: Union[list[int], np.ndarray] = None,
    temp_col: Optional[str] = None,
    ds_target: int = 250,
    acc: Optional[pd.DataFrame] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Downsample pre-processed data and any acceleration data for
    quicker plot rendering on the dashboard."""
    def __decimate(y: np.ndarray) -> np.ndarray:
        """Helper function for zero-phase anti-alias filtering and decimation."""
        if ds_factor == 1:
            return y
        cutoff = min(0.45 / ds_factor, 0.49)
        b = firwin(numtaps = 129, cutoff = cutoff)
        y_f = filtfilt(b, [1.0], y, method = "pad",
                       padlen = min(3 * max(len(b), 1), len(y) - 1)) \
            if len(y) > 10 else y
        return y_f[::ds_factor]

    # Validate column inputs
    if signal_type not in df.columns:
        raise KeyError(f'{signal_type} not found in input DataFrame.')
    if temp_col is not None and temp_col not in df.columns:
        raise KeyError(f'{temp_col} not found in input DataFrame.')

    # Choose x and y columns
    x_col = 'Timestamp' if 'Timestamp' in df.columns else 'Sample'
    y_col = 'Filtered' if 'Filtered' in df.columns else signal_type

    # Calculate downsampling factor
    ds_factor = max(1, int(fs) // ds_target)

    if ds_factor != 1:
        ds_fs = int(fs / ds_factor)
        ds_idx = np.arange(0, len(df), ds_factor)

        # Decimate primary signal
        y_dec = __decimate(df[y_col])
        ds = pd.DataFrame({x_col: df[x_col].iloc[ds_idx].to_numpy(), y_col: y_dec})

        # Rescale detected, artifactual, and corrected beat indices
        if beats_ix is not None:
            down_beats = np.rint(
                beats_ix / ds_factor).astype(int).clip(0, len(ds) - 1)
            ds.loc[down_beats, 'Beat'] = 1
        if artifacts_ix is not None:
            down_artifacts = np.rint(
                artifacts_ix / ds_factor).astype(int).clip(0, len(ds) - 1)
            ds.loc[down_artifacts, 'Artifact'] = 1
        if corrected_beats_ix is not None:
            down_corrected_beats = np.rint(
                corrected_beats_ix / ds_factor).astype(int).clip(0, len(ds) - 1)
            ds.loc[down_corrected_beats, 'Corrected'] = 1

        # Downsample acceleration data
        ds_acc = None
        if acc is not None:
            acc_dec = __decimate(acc['Magnitude'])
            ds_acc = pd.DataFrame(
                {x_col: df[x_col].iloc[ds_idx].to_numpy(),
                 'Magnitude': acc_dec})

        # Downsample IBI data for cardiac signals
        ds_ibi, ds_ibi_corrected = None, None
        if signal_type in ('ECG', 'PPG', 'BVP'):
            ds_ibi = compute_ibis(ds, ds_fs, down_beats, ts_col = x_col)
            if corrected_beats_ix is not None:
                ds_ibi_corrected = compute_ibis(
                    ds, ds_fs, down_corrected_beats, ts_col = x_col)

            # Downsample optional TEMP data for EDA signal
            if temp_col is not None:
                ds['TEMP'] = __decimate(df[temp_col])
            return ds, ds_ibi, ds_ibi_corrected, ds_acc, ds_fs
        else:
            return ds, ds_ibi, ds_ibi_corrected, ds_acc, ds_fs

    else:
        ibi, ibi_corrected = None, None
        if signal_type in ('ECG', 'PPG', 'BVP'):
            ibi = compute_ibis(df, fs, beats_ix, ts_col = x_col)
            if corrected_beats_ix is not None:
                ibi_corrected = compute_ibis(
                    df, fs, corrected_beats_ix, ts_col = x_col)
        return df, ibi, ibi_corrected, acc, fs


def _cardiac_summary_table(sqa_df: pd.DataFrame) -> dbc.Table:
    """Display the cardiac SQA summary table."""

    # Calculate average heart rate
    valid_df = sqa_df[sqa_df.Invalid != 1].copy().reset_index(drop = True)
    valid_ix = np.where(np.diff(valid_df['N Detected']) < 10)[0]
    valid_df = valid_df.loc[valid_ix].reset_index(drop = True)
    avg_n = '{0:.2f}'.format(valid_df['N Detected'].mean())
    missing_n = len(sqa_df.loc[sqa_df['N Missing'] > 0])
    artifact_n = len(sqa_df.loc[sqa_df['N Artifact'] > 0])
    invalid_n = len(sqa_df.loc[sqa_df['Invalid'] == 1])
    invalid_prop = '{0:.2f}%'.format(
        (invalid_n / sqa_df['Segment'].max()) * 100)
    avg_missing = '{0:.2f}%'.format(sqa_df['% Missing'].mean())
    avg_artifact = sqa_df.loc[sqa_df['% Artifact'] > 0, '% Artifact'].mean()

    # Set NaN average artifact values to zero
    if pd.isna(avg_artifact):
        avg_artifact = 0
    avg_artifact = f'{avg_artifact:.2f}%'

    # Build summary table data
    data = [
        ('Average Number of Beats', avg_n),
        ('Segments with Missing Beats', missing_n),
        ('Segments with Artifactual Beats', artifact_n),
        ('Segments with Invalid Beats', invalid_n),
        ('% Invalid Data', invalid_prop),
        ('Average % Missing Beats/Segment', avg_missing),
        ('Average % Artifactual Beats/Segment', avg_artifact)
    ]

    # Wrap in a dbc.Table
    rows = [
        html.Tr([
            html.Td(label),
            html.Td(value)
        ]) for label, value in data
    ]
    table = dbc.Table(
        rows,
        className = 'segmentTable',
        striped = False,
        bordered = False,
        hover = False
    )

    return table, data

def _eda_summary_table(
    sqa_df: pd.DataFrame,
    tonic_scl: np.ndarray,
    scr_series: Optional[np.ndarray] = None,
    seg_size: Optional[int] = None
) -> dbc.Table:
    """Display the EDA SQA summary table."""

    if scr_series is not None:
        scr_peaks = np.nan_to_num(scr_series, nan = 0)
        n_seg = int(np.ceil(len(scr_peaks)) / seg_size)
        scr_segments = scr_peaks[:n_seg * seg_size].reshape(n_seg, seg_size)
        avg_scr_seg = round(scr_segments.sum(axis = 1).mean(), 2)
    else:
        avg_scr_seg = 'N/A'

    med_scl = round(np.median(tonic_scl), 2)
    invalid_n = len(sqa_df.loc[sqa_df['N Invalid'] > 0])
    invalid_prop = '{0:.2f}%'.format(sqa_df['% Invalid'].mean())
    oor_prop = '{0:.2f}%'.format(sqa_df['% Out of Range'].mean())
    excess_slope_prop = '{0:.2f}%'.format(sqa_df['% Excessive Slope'].mean())
    temp_oor_mean = sqa_df['% Temp Out of Range'].mean()
    if pd.isna(temp_oor_mean):
        temp_oor_prop = 'N/A'
    else:
        temp_oor_prop = f'{temp_oor_mean:.2f}%'

    # Build summary table data
    data = [
        ('Median Tonic SCL', med_scl),
        ('Average SCR Peaks/Segment', avg_scr_seg),
        ('Segments with Invalid Data', invalid_n),
        ('Average % Invalid Data', invalid_prop),
        ('Average % Out of Range', oor_prop),
        ('Average % Excessive Slope', excess_slope_prop),
        ('Average % Temp Out of Range', temp_oor_prop)
    ]

    # Wrap in a dbc.Table
    rows = [
        html.Tr([
            html.Td(label),
            html.Td(value)
        ]) for label, value in data
    ]
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