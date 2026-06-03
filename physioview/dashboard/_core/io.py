"""
Data I/O and processing utilities for the PhysioView Dashboard.

This module provides functionality for file validation, data parsing,
configuration file management, and other data operations.

All functions in this module are intended for internal use by the dashboard
and should not be called directly from external code.
"""

from typing import Callable, Optional, Union
from os import path
from pathlib import Path
from io import BytesIO, StringIO
from scipy.signal import filtfilt, firwin
from physioview.physioview import compute_ibis
from time import sleep
import base64
import json
import numpy as np
import pandas as pd
import pyedflib
import zipfile

# ================================ VALIDATION ================================
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

# ====================== CONFIGURATION FILE MANAGEMENT =======================
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
    scr_detector: str,
    scr_amp: float,
    headers: Optional[dict] = None,
    temp_on: bool = False,
    temp_var: Optional[str] = None,
    eda_min: Optional[float] = None,
    eda_max: Optional[float] = None
) -> str:
    """Create a JSON-formatted configuration file of user SQA parameters."""

    # Save user configuration
    configs = {'source': source,
               'data type': dtype,
               'sampling rate': fs,
               'segment size': seg_size,
               'filters': filter_on,
               'scr detector': scr_detector,
               'scr amplitude': scr_amp,
               'artifact identification method': artifact_method,
               'artifact tolerance': artifact_tol,
               'use temperature': temp_on,
               'temperature variable': temp_var,
               'minimum eda': eda_min,
               'maximum eda': eda_max}

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

# ============================= DATA OPERATIONS ==============================
def _get_csv_headers(csv: str) -> list[str]:
    """Get the headers of a user-uploaded CSV file in a list."""
    initial = pd.read_csv(csv, nrows = 1)
    headers = initial.columns.tolist()
    return headers

def _validate_event_file_ext(filename: str) -> bool:
    """Check for valid event file extensions."""
    ext = path.splitext(filename)[1].lower()
    if ext not in ('.csv', '.txt', '.zip'):
        return False
    return True

def _decode_bytes(raw_bytes: bytes) -> str:
    """Decode bytes to str, handling UTF-16, UTF-8, and Latin-1 encodings."""
    if raw_bytes.startswith((b'\xff\xfe', b'\xfe\xff')):
        return raw_bytes.decode('utf-16')
    try:
        decoded = raw_bytes.decode('utf-8-sig')
        if '\x00' in decoded:
            return raw_bytes.decode('utf-16-le')
        return decoded
    except UnicodeDecodeError:
        return raw_bytes.decode('latin-1')

def _parse_event_data(
    contents: str,
    filename: str
) -> Union[dict, pd.DataFrame]:
    """Parse event timestamps uploaded with dcc.Upload component."""
    content_type, content_string = contents.split(',')
    raw = base64.b64decode(content_string)

    if filename.endswith('.zip'):
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            csv_files = [f for f in zf.namelist()
                         if f.endswith(('.csv', '.txt'))
                         and not f.endswith('/')
                         and not f.startswith('__MACOSX/')
                         and '/._' not in f
                         and not f.endswith('.DS_Store')]
            if not csv_files:
                return {}
            event_data = {}
            for csv_file in csv_files:
                with zf.open(csv_file) as f:
                    raw_bytes = f.read()
                event_df = pd.read_csv(
                    StringIO(_decode_bytes(raw_bytes)),
                    sep = None,
                    engine = 'python',
                    skipinitialspace = True)
                # Replace underscores with spaces
                event_data['event'] = event_data['event'].str.replace('_', ' ')
                key = path.splitext(path.basename(csv_file))[0]
                event_data[key] = event_df
    else:
        event_data = pd.read_csv(
            StringIO(_decode_bytes(raw)),
            sep = None,
            engine = 'python',
            skipinitialspace = True)
        # Replace underscores with spaces
        event_data['event'] = event_data['event'].str.replace('_', ' ')
    
    return event_data

def _parse_temp_csv(contents: str) -> pd.DataFrame:
    """Parse temperature data uploaded with dcc.Upload component."""
    content_type, content_string = contents.split(',')
    raw = base64.b64decode(content_string)
    buf = StringIO(raw.decode('utf-8'))
    return pd.read_csv(buf)

def _convert_timestamps(ts: pd.Series) -> pd.Series:
    """Convert a Series of timestamps to tz-naive UTC datetime64[ns]. Handles
    Unix timestamps (seconds or milliseconds) and ISO8601 strings."""
    unix_format = _check_unix(ts)
    if unix_format is not None:
        converted = pd.to_datetime(ts, unit = unix_format)
        return converted

    # Parse ISO8601 strings
    converted = pd.to_datetime(ts, errors = 'coerce', format = 'ISO8601')
    if converted.isna().any():
        raise ValueError(
            'Invalid timestamp format detected. Please ensure '
            'timestamps are in a valid datetime format.')

    # Convert to timezone-naive UTC
    if converted.dt.tz is not None:
        converted = converted.dt.tz_convert('UTC').dt.tz_localize(None)
    return converted

def _setup_data(
    csv: str,
    dtype: str,
    dropdowns: list[str],
    temp_var: Optional[str] = None,
    event_data: Optional[pd.DataFrame] = None,
    has_ts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and map columns of uploaded CSV data to variables."""
    if not has_ts and event_data is not None:
        raise ValueError(
            'Cannot segment by events without timestamps in your data. '
            'Please upload data with a timestamp column or disable '
            'event-based segmentation.')

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
        rename_map[temp_var] = 'Temp'
    df.rename(columns = rename_map, inplace = True)

    # Convert timestamps to datetime format
    if has_ts:
        df['Timestamp'] = _convert_timestamps(df['Timestamp'])
        ts_col = 'Timestamp'

    # Insert 'Sample column' if no timestamp
    else:
        df.insert(0, 'Sample', np.arange(len(df)) + 1)
        ts_col = 'Sample'

    # Build signal DataFrame with 'TEMP' if it exists
    data_cols = [ts_col, dtype]
    if 'temp' in df.columns.str.lower():
        data_cols.append('Temp')
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

    def __ix_to_pos(
        ix: Union[list[int], np.ndarray],
        index: pd.Index
    ) -> np.ndarray:
        """Convert df index labels to positional indices; drop missing labels."""
        ix = np.asarray(ix, dtype = int)
        if ix.size == 0:
            return ix
        pos = index.get_indexer(ix)
        return pos[pos >= 0]

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

        # Convert to positional indices
        beats_ix = __ix_to_pos(beats_ix, df.index)
        artifacts_ix = __ix_to_pos(artifacts_ix, df.index)
        if corrected_beats_ix is not None:
            corrected_beats_ix = __ix_to_pos(corrected_beats_ix, df.index)

        # Rescale detected, artifactual, and corrected beat indices
        down_beats = np.rint(
            beats_ix / ds_factor).astype(int).clip(0, len(ds) - 1)
        ds.loc[down_beats, 'Beat'] = 1
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

            # Downsample optional temperature data for EDA signal
            if temp_col is not None:
                ds['Temp'] = __decimate(df[temp_col])
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

def _create_render(
    render_subdir: str,
    ds_data: Optional[pd.DataFrame] = None,
    ds_ibi: Optional[pd.DataFrame] = None,
    ds_acc: Optional[pd.DataFrame] = None,
    ds_ibi_corrected: Optional[pd.DataFrame] = None
) -> None:
    """
    Write downsampled data to the 'temp/_render' subdirectory.

    Parameters
    ----------
    render_subdir : str
        The name of the subfolder within the 'temp/_render' subdirectory,
        e.g., 'baseline' for a Baseline event window.
    ds_data : pd.DataFrame, optional
        The downsampled physiological data for rendering.
    ds_ibi : pd.DataFrame, optional
        Any downsampled IBI data for rendering.
    ds_acc: pd.DataFrame, optional
        Any downsampled accelerometer data for rendering.
    ds_ibi_corrected: pd.DataFrame, optional
        Any downsampled corrected IBI data for rendering.
    """

    # Create subfolder within '_render'
    root = Path(__file__).resolve().parents[3]
    render_dir = root / 'temp' / '_render'
    render_subdir = render_dir / render_subdir
    render_subdir.mkdir(parents = True, exist_ok = True)

    if ds_data is not None:
        ds_data.to_csv(render_subdir / 'signal.csv', index = False)
    if ds_ibi is not None:
        ds_ibi.to_csv(render_subdir / 'ibi.csv', index = False)
    if ds_acc is not None:
        ds_acc.to_csv(render_subdir / 'acc.csv', index = False)
    if ds_ibi_corrected is not None:
        ds_ibi_corrected.to_csv(render_subdir / 'ibi_corrected.csv',
                                index = False)

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
        with zipfile.ZipFile(zip_path, 'w') as archive:
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
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, file_path in enumerate(files):
            file_name = file_path.name
            with open(file_path, 'rb') as f:
                zf.writestr(file_name, f.read())

            # If set_progress, update progress
            if set_progress is not None:
                remaining = max(total_progress - progress_start, 0)
                frac = (i + 1) / n_files
                progress = (progress_start + remaining
                            * frac) / total_progress * 100
                set_progress((progress, f'{progress:.0f}%'))
                sleep(0.5)
    out.seek(0)
    return out