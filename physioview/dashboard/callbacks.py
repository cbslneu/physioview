from . import _core
from typing import Literal, Optional, Tuple
from dash import html, Input, Output, State, ctx, callback, no_update
from dash.exceptions import PreventUpdate
from dash.dcc import send_bytes
from physioview import physioview
from physioview.pipeline import ACC, SQA
from physioview.pipeline.EDA import compute_tonic_scl, compute_features
from flirt.hrv import get_hrv_features
from pathlib import Path
from time import sleep
from io import BytesIO
from datetime import datetime
from collections import defaultdict
import dash_uploader as du
import zipfile
import shutil
import pandas as pd
import numpy as np
import traceback

# Define local paths for outputs
root = Path(__file__).resolve().parents[2]
temp_path = root / 'temp'
render_dir = temp_path / '_render'
beat_editor_dir = root / 'beat-editor'

# TODO: Refactor into a separate module with the run_pipeline() long callback
def _preprocess_cardiac_by_event(
    preprocessor: _core.Preprocessor,
    data: pd.DataFrame,
    fs: int,
    dtype: Literal['ECG', 'PPG'],
    fname: str,
    acc: Optional[pd.DataFrame] = None,
    ts_col: Optional[str] = None,
    artifact_method: Optional[str] = None,
    artifact_tol: Optional[float] = None
) -> dict[str, float]:
    """
    Run event-based cardiac preprocessing and write outputs.

    Parameters
    ----------
    preprocessor : _core.Preprocessor
        The initialized preprocessor object with event data.
    data : pd.DataFrame
        A DataFrame containing the raw cardiac signal to preprocess.
    fs : int
        The sampling rate of the input cardiac data in Hz.
    dtype : str
        The signal type label for output filenames. Must be either 'ECG' or
        'PPG'.
    fname : str
        The file stem used to prefix output filenames.
    acc : pd.DataFrame or None
        A DataFrame containing preprocessed accelerometer data, if any.
    ts_col : str or None
        The name of the timestamp column, if any.
    artifact_method : str
        The selected artifact detection method, given by the value of the
        'artifact-method' dcc.Dropdown.
    artifact_tol : float
        The artifact tolerance threshold, given by the value of the
        'artifact-tol' dcc.Input.

    Returns
    -------
    event_durations : dict[str, float]
        A dictionary mapping '{fname}_{event_label}' to duration in seconds.

    Raises
    ------
    RuntimeError
        If preprocessing fails.
    """
    try:
        preprocessed, preprocessed_by_event, metrics_by_event = \
            preprocessor.preprocess_event(
                data, artifact_method = artifact_method,
                artifact_tol = artifact_tol)
    except Exception:
        print(traceback.format_exc())
        raise RuntimeError('Event-based cardiac preprocessing failed.')

    event_durations = {}
    ds_fs = None
    for event_label, event_data in preprocessed_by_event.items():
        event_durations[f'{fname}_{event_label}'] = len(event_data) / fs

        # Write event data to 'temp' folder
        event_data.to_csv(
            temp_path / f'{fname}_{event_label}_{dtype}.csv',
            index = False)

        # Compute IBIs
        beats_ix = preprocessor.peaks_by_event[event_label]
        artifacts_ix = preprocessor.artifacts_by_event[event_label]
        event_ibi = physioview.compute_ibis(
            event_data, fs, beats_ix, ts_col = ts_col)
        event_ibi.to_csv(
            temp_path / f'{fname}_{event_label}_IBI.csv',
            index = False)

        # Downsample for rendering
        ds_data, ds_ibi, _, ds_acc, ds_fs = \
            _core.io._downsample_data(
                event_data, fs, dtype, beats_ix, artifacts_ix,
                acc = acc.loc[event_data.index] if acc is not None
                else None)

        # Write downsampled event data to '_render'
        _core.io._create_render(
            f'{fname}_{event_label}', ds_data, ds_ibi, ds_acc)

    # Write SQA metrics to 'temp' folder
    for event_label, metrics in metrics_by_event.items():
        metrics.to_csv(
            temp_path / f'{fname}_{event_label}_SQA.csv',
            index = False)

    return event_durations, ds_fs


def _preprocess_eda_by_event(
    preprocessor: _core.Preprocessor,
    data: pd.DataFrame,
    fname: str,
    acc: Optional[pd.DataFrame] = None,
    rs: Optional[int] = None,
    min_peak_amp: Optional[float] = None,
    temp: Optional[np.ndarray] = None,
    eda_min: Optional[float] = None,
    eda_max: Optional[float] = None
) -> dict[str, float]:
    """
    Run event-based EDA preprocessing and write outputs.

    Parameters
    ----------
    preprocessor : _core.Preprocessor
        The initialized preprocessor object with event data.
    data : pd.DataFrame
        A DataFrame containing the raw EDA signal to preprocess.
    fs : int
        The sampling rate of the input EDA data in Hz.
    fname : str
        The file stem used to prefix output filenames.
    acc : pd.DataFrame or None
        A DataFrame containing preprocessed accelerometer data, if any.
    ts_col : str or None
        The name of the timestamp column, if any.
    rs : int or None
        An optional target resampling rate in Hz, given by the value of the
        'resampling-rate' dcc.Input.
    min_peak_amp : float or None
        The minimum SCR peak amplitude threshold, given by the value of the
        'scr-amp-thresh' dcc.Input.
    temp : np.ndarray or None
        An array containing the skin temperature data, if any.
    eda_min : float or None
        The minimum valid EDA value in microsiemens, given by the value of
        the 'eda-valid-min' dcc.Input.
    eda_max : float or None
        The maximum valid EDA value in microsiemens, given by the value of
        the 'eda-valid-max' dcc.Input.

    Returns
    -------
    event_durations : dict[str, float]
        A dictionary mapping '{fname}_{event_label}' to duration in seconds.

    Raises
    ------
    RuntimeError
        If preprocessing fails.
    """
    try:
        preprocessed, preprocessed_by_event, metrics_by_event = \
            preprocessor.preprocess_event(
                data, rs, min_peak_amp, temp_data = temp,
                eda_min = eda_min, eda_max = eda_max)
    except Exception:
        print(traceback.format_exc())
        raise RuntimeError('Event-based EDA preprocessing failed.')

    event_durations = {}
    for event_label, event_data in preprocessed_by_event.items():
        event_durations[f'{fname}_{event_label}'] = \
            len(event_data) / preprocessor.fs

        # Write event data to 'temp' folder
        event_data.to_csv(
            temp_path / f'{fname}_{event_label}_EDA.csv',
            index = False)

        # Write downsampled event data to '_render'
        _core.io._create_render(
            f'{fname}_{event_label}', event_data, ds_acc = acc)

    # Write SQA metrics to 'temp' folder
    for event_label, metrics in metrics_by_event.items():
        metrics.to_csv(
            temp_path / f'{fname}_{event_label}_SQA.csv',
            index = False)

    return event_durations, preprocessor.fs


def get_callbacks(app):
    """Attach callback functions to the dashboard app."""

    # ============================= DATA UPLOAD ===============================
    du.configure_upload(app, str(temp_path), use_upload_id = True)
    @du.callback(
        output = [
            Output('file-check', 'children'),
            Output('run-data', 'disabled'),
            Output('configure', 'disabled'),
            Output('e4-data-type-container', 'hidden'),  # E4 data types div
            Output('memory-load', 'data'),
        ],
        id = 'dash-uploader'
    )
    def db_get_file_types(filenames):
        """Save the data type to the local memory depending on the file
        type."""
        if not filenames:
            return [[], True, True, True, None]

        session_path = Path(filenames[0]).parent
        filename = filenames[0]

        # Default visibility
        disable_run = True
        disable_configure = True
        hide_e4_dtypes = True

        ext = filenames[0].lower().rsplit('.', 1)[-1]
        if ext == 'edf':
            if _core.io._check_edf(filenames[0]) == 'ECG':
                file_check = [
                    html.I(className = 'fa-solid fa-circle-check',
                           style = {'color': '#63e6be', 'marginRight': '5px'}),
                    html.Span('Data loaded.')
                ]
                data = {'source': 'Actiwave',
                        'filename': filenames[0]}
                disable_run = False
                disable_configure = False
            else:
                file_check = [
                    html.I(className = 'fa-solid fa-circle-xmark'),
                    html.Span('Invalid data type!')]
                data = 'invalid'

        # Zip is either an Empatica E4 or batch file
        elif ext == 'zip':
            z = zipfile.ZipFile(filename)

            # Check if Empatica E4 data
            empatica_files = ['ACC.csv',
                              'EDA.csv',
                              'BVP.csv',
                              'TEMP.csv',
                              'IBI.csv',
                              'HR.csv',
                              'info.txt',
                              'tags.csv']
            if all(f in z.namelist() for f in empatica_files):
                file_check = [
                    html.I(className = 'fa-solid fa-circle-check',
                           style = {'color': '#63e6be',
                                    'marginRight': '5px'}),
                    html.Span('Data loaded.')
                ]
                data = {'source': 'E4',
                        'filename': filename}
                disable_run = False
                disable_configure = False
                hide_e4_dtypes = False

            # Check if batch data
            else:
                # Filter out metadata from zip file
                zfiles = [f.split('/', 1)[1] for f in z.namelist()
                          if '/' in f and not f.startswith('__MACOSX/') and
                          not f.endswith('.DS_Store') and '/._' not in f and
                          not f.endswith('/')]

                if all(f.endswith('.csv') for f in zfiles):
                    file_check = [
                        html.I(className = 'fa-solid fa-circle-check',
                               style = {'color': '#63e6be',
                                        'marginRight': '5px'}),
                        html.Span('Data loaded.')
                    ]

                    # Clear stale batch CSVs from previous uploads in the same
                    # session folder
                    batch_dir = session_path / 'batch'
                    if batch_dir.exists():
                        shutil.rmtree(batch_dir)
                    batch_dir.mkdir(parents = True, exist_ok = True)
                    data = {'source': 'batch',
                            'filename': filename}
                    disable_run = False
                    disable_configure = False
                else:
                    data = 'invalid'
                    file_check = [
                        html.I(className = 'fa-solid fa-circle-xmark'),
                        html.Span('Invalid data type!')
                    ]

        # Check if single CSV file
        elif ext == 'csv':
            file_check = [
                html.I(className = 'fa-solid fa-circle-check',
                       style = {'color': '#63e6be', 'marginRight': '5px'}),
                html.Span('Data loaded.')
            ]
            data = {'source': 'csv',
                    'filename': filename}
            disable_run = False
            disable_configure = False

        else:
            return [[], True, True, True, None]

        # Clear Beat Editor directories
        _core.startup._clear_edits()

        # Clear stale files in 'temp' directory
        for p in temp_path.iterdir():
            if p == session_path or p == render_dir:
                continue
            if p.is_file() or p.is_symlink():
                p.unlink()
            else:
                shutil.rmtree(p, ignore_errors = True)

        return [file_check, disable_run, disable_configure,
                hide_e4_dtypes, data]

    # ==================== ENABLE CONFIGURATION UPLOAD ========================
    # === Toggle configuration uploader =======================================
    @app.callback(
        Output('config-upload-div', 'hidden'),
        Input('toggle-config', 'on'),
        prevent_initial_call = True
    )
    def db_enable_config_upload(toggle_on):
        """Display configuration file upload."""
        if toggle_on is True:
            hidden = False
        else:
            hidden = True
        return hidden

    # === Read JSON configuration file ========================================
    @du.callback(
        output = Output('config-memory', 'data'),
        id = 'config-uploader'
    )
    def db_get_config_file(cfg_file):
        configs = _core.io._load_config(cfg_file[0])
        return configs

    # ======================== ENABLE DATA PARAMETERS =========================
    @app.callback(
        [Output('sampling-rate', 'value', allow_duplicate = True),
         Output('resample', 'hidden'),
         Output('resampling-rate', 'disabled'),
         Output('load-temperature', 'hidden'),
         Output('temp-upload-section', 'hidden'),
         Output('preprocess-data', 'hidden', allow_duplicate = True),
         Output('beat-detector-settings', 'hidden'),
         Output('artifact-settings', 'hidden'),
         Output('eda-preprocessing', 'hidden', allow_duplicate = True),
         Output('select-scr-detector', 'hidden', allow_duplicate = True),
         Output('scr-amplitude-threshold', 'hidden'),
         Output('beat-detectors', 'options', allow_duplicate = True),
         Output('beat-detectors', 'value', allow_duplicate = True),
         Output('scr-detectors', 'options'),
         Output('scr-detectors', 'value', allow_duplicate = True),
         Output('seg-size', 'value', allow_duplicate = True)],
        [Input('e4-data-types', 'value'),
         Input('data-types', 'value'),
         Input('toggle-resample', 'on'),
         Input('toggle-temp-data', 'on'),
         State('memory-load', 'data')],
        prevent_initial_call = True
    )
    def db_enable_dtype_specific_parameters(e4_dtype, dtype, toggle_rs_on,
                                            toggle_temp_on, loaded_data):
        """Enable parameters specific to data types of CSV sources."""
        load_temp_hidden = True
        temp_upload_hidden = True
        preprocess_data_hidden = True
        eda_preprocess_hidden = True
        beat_detector_settings_hidden = True
        artifact_settings_hidden = True
        scr_amp_thresh_hidden = True
        resample_hidden = True
        resample_disabled = True
        beat_detectors = no_update
        default_beat_detector = no_update
        scr_detectors = []
        default_scr_detector = None
        data_source = loaded_data['source']
        seg_size = 60
        fs = 500

        # Handle EDA components
        if dtype == 'EDA' or e4_dtype == 'EDA':
            resample_hidden = False
            load_temp_hidden = False
            eda_preprocess_hidden = False
            preprocess_data_hidden = False
            seg_size = 180
            if toggle_rs_on is True:
                resample_disabled = False
            if toggle_temp_on is True:
                temp_upload_hidden = False
            scr_detectors = [
                {'label': 'Nabian et al. (2018)', 'value': 'nabian'},
                {'label': 'Threshold-Based', 'value': 'threshold'}
            ]
            default_scr_detector = 'threshold'
            scr_amp_thresh_hidden = False
            fs = 4

        # Handle cardiac components
        trig = ctx.triggered_id
        if trig == 'e4-data-types' and e4_dtype == 'PPG':
            fs = 64
            eda_preprocess_hidden = True
            preprocess_data_hidden = False
            beat_detector_settings_hidden = False
            artifact_settings_hidden = False
            beat_detectors = [
                {'label': 'Elgendi et al. (2013)', 'value': 'erma'},
                {'label': 'Van Gent et al. (2018)', 'value': 'adaptive_threshold'}]
            default_beat_detector = 'adaptive_threshold'
        elif ctx.triggered_id == 'data-types' and dtype in ('PPG', 'ECG'):
            fs = 500
            eda_preprocess_hidden = True
            preprocess_data_hidden = False
            beat_detector_settings_hidden = False
            artifact_settings_hidden = False
            if dtype == 'PPG':
                beat_detectors = [
                    {'label': 'Elgendi et al. (2013)', 'value': 'erma'},
                    {'label': 'Van Gent et al. (2018)', 'value': 'adaptive_threshold'}]
                default_beat_detector = 'adaptive_threshold'
            else:
                beat_detectors = [
                    {'label': 'Manikandan & Soman (2012)', 'value': 'manikandan'},
                    {'label': 'Engels & Zeelenberg (1979)', 'value': 'engzee'},
                    {'label': 'Nabian et al. (2018)', 'value': 'nabian'},
                    {'label': 'Pan & Tompkins (1985)', 'value': 'pantompkins'}]
                default_beat_detector = 'manikandan'

        return [fs, resample_hidden, resample_disabled,
                load_temp_hidden, temp_upload_hidden, preprocess_data_hidden,
                beat_detector_settings_hidden, artifact_settings_hidden,
                eda_preprocess_hidden,   # eda-preprocessing
                eda_preprocess_hidden,   # select-scr-detector
                scr_amp_thresh_hidden,
                beat_detectors, default_beat_detector,
                scr_detectors, default_scr_detector, seg_size]

    # === Toggle event segmentation settings ==================================
    @app.callback(
        [Output('event-segmentation-options', 'disabled'),
         Output('event-file-upload-div', 'hidden')],
        Input('toggle-event-segmentation', 'on'),
        prevent_initial_call = True
    )
    def toggle_data_segmentation(toggle_on):
        """Toggle the event segmentation settings."""
        if toggle_on is True:
            return False, False
        else:
            return True, True

    # === Set windowed/entire event segmentation ==============================
    @app.callback(
        [Output('peak-detection-mode', 'options'),
         Output('peak-detection-mode', 'value'),
         Output('seg-size', 'disabled'),
         Output('seg-size', 'value'),
         Output('segment-data-by-time', 'style')],
        [Input('event-segmentation-options', 'value'),
         Input('seg-size', 'value'),
         Input('toggle-event-segmentation', 'on')],
        [State('peak-detection-mode', 'value')],
        prevent_initial_call = True
    )
    def handle_segmentation_params(segment_event_by, seg_size, toggle_on,
                                   current_mode):
        """Enable or disable the segment size input and 'By Segment' peak detection
        option based on the event segmentation toggle state and selected segmentation
        mode, greying out segment-related controls when entire-event processing is
        active."""
        disable = segment_event_by == 'entire' and toggle_on
        segment_options = lambda disabled: [
            {'label': 'Entire Signal', 'value': 'entire'},
            {'label': 'By Segment', 'value': 'segment', 'disabled': disabled}
        ]
        if disable:
            return segment_options(True), 'entire', True, None, \
                {'color': '#bababa', 'fontStyle': 'italic'}
        return segment_options(False), current_mode or 'entire', False, \
            seg_size or 60, {}

    # === Open advanced filter cutoff settings ================================
    @app.callback(
        [Output('filter-config-btn', 'hidden'),
         Output('filter-customization-modal', 'is_open'),
         Output('dtype-validator', 'is_open')],
        [Input('toggle-filter', 'on'),
         Input('filter-config-btn', 'n_clicks')],
        [State('memory-load', 'data'),
         State('data-types', 'value'),
         State('e4-data-types', 'value')],
        prevent_initial_call = True
    )
    def handle_filter_config_link(filter_on, n_clicks, data, dtype, e4_dtype):
        """Enable/disable the filter settings link based on the filter
        toggle state and display/hide the settings when the link is clicked."""
        trig = ctx.triggered_id

        # Handle filter toggle
        if trig == 'toggle-filter':
            if filter_on:
                # Filter enabled; keep settings hidden
                return False, False, False
            else:
                # Filter disabled; hide link and settings
                return True, False, False

        # Handle link click only if the filter toggle is already on
        if trig == 'filter-config-btn':
            if filter_on:
                if data['source'] == 'Actiwave':
                    return False, True, False
                elif data['source'] == 'E4':
                    if e4_dtype is None:
                        return False, False, True
                    else:
                        return False, True, False
                else:
                    if dtype is None:
                        return False, False, True
                    else:
                        return False, True, False

        # Otherwise, keep the link and settings hidden
        return True, False, False

    # === Set filter parameters ============================================
    @app.callback(
        [Output('lower-cutoff-div', 'hidden'),
         Output('filter-lowcut', 'value', allow_duplicate = True),
         Output('filter-highcut', 'value', allow_duplicate = True),
         Output('filter-order-div', 'hidden'),
         Output('filter-order', 'value', allow_duplicate = True),
         Output('filter-rp-div', 'hidden'),
         Output('filter-rp', 'value', allow_duplicate = True),
         Output('filter-rs-div', 'hidden'),
         Output('filter-rs', 'value', allow_duplicate = True),
         Output('filter-window-len-div', 'hidden'),
         Output('filter-window-len', 'value', allow_duplicate = True),
         Output('filter-length-div', 'hidden'),
         Output('filter-length', 'value', allow_duplicate = True),
         Output('filter-window-type-div', 'hidden'),
         Output('filter-window-type', 'value', allow_duplicate = True),
         Output('selected-filter', 'children')],
        [Input('memory-load', 'data'),
         Input('data-types', 'value'),
         Input('e4-data-types', 'value'),
         Input('beat-detectors', 'value'),
         Input('cancel-config-btn', 'n_clicks'),
         Input('reset-to-default-btn', 'n_clicks')],
        prevent_initial_call = True
    )
    def set_default_filter_params(data, dtype, e4_dtype, beat_detector, n_cancel, n_reset):
        """Populate and show/hide filter parameter inputs based on the
        selected data type and beat detector."""

        if data is None:
            raise PreventUpdate

        if data['source'] == 'Actiwave':
            selected_dtype = 'ECG'
        else:
            selected_dtype = dtype if dtype in ['ECG', 'PPG', 'EDA'] else e4_dtype
            if selected_dtype is None:
                raise PreventUpdate

        filter_params = _core.Preprocessor.DEFAULT_FILTER_PARAMS[selected_dtype]
        if selected_dtype == 'ECG':
            if beat_detector not in filter_params:
                raise PreventUpdate
            filter_params = filter_params[beat_detector]
        lowcut = filter_params.get('lowcut')
        highcut = filter_params.get('highcut')
        order = filter_params.get('order')
        rp = filter_params.get('rp')
        rs = filter_params.get('rs')
        window_len = filter_params.get('window_len')
        filter_length = filter_params.get('filter_length')
        window_type = filter_params.get('window_type')
        filt_type = filter_params.get('filt_type', 'No filter selected.')

        hide_lowcut = True if lowcut is None else False
        hide_order = True if order is None else False
        hide_rp = True if rp is None else False
        hide_rs = True if rs is None else False
        hide_window_len = True if window_len is None else False
        hide_filter_length = True if filter_length is None else False
        hide_window_type = True if window_type is None else False

        return [hide_lowcut, lowcut, highcut, hide_order, order,
                hide_rp, rp, hide_rs, rs, hide_window_len, window_len,
                hide_filter_length, filter_length,
                hide_window_type, window_type, filt_type]

    # === Close filter customization modal =====================================
    @app.callback(
        [Output('filter-customization-modal', 'is_open', allow_duplicate = True),
         Output('empty-param-error-div', 'hidden'),
         Output('lowcut-highcut-error-div', 'hidden')],
        [Input('apply-filter-btn', 'n_clicks'),
         Input('cancel-config-btn', 'n_clicks'),
         Input('reset-to-default-btn', 'n_clicks')],
        [State('lower-cutoff-div', 'hidden'),
         State('filter-lowcut', 'value'),
         State('upper-cutoff-div', 'hidden'),
         State('filter-highcut', 'value'),
         State('filter-order-div', 'hidden'),
         State('filter-order', 'value'),
         State('filter-rp-div', 'hidden'),
         State('filter-rp', 'value'),
         State('filter-rs-div', 'hidden'),
         State('filter-rs', 'value'),
         State('filter-window-len-div', 'hidden'),
         State('filter-window-len', 'value'),
         State('filter-length-div', 'hidden'),
         State('filter-length', 'value'),
         State('filter-window-type-div', 'hidden'),
         State('filter-window-type', 'value')],
        prevent_initial_call = True
    )
    def close_filter_customization_modal(n_apply, n_cancel, n_reset, hide_lowcut, lowcut, hide_highcut, highcut, \
        hide_order, order, hide_rp, rp, hide_rs, rs, hide_window_len, window_len, hide_filter_length, \
            filter_length, hide_window_type, window_type):
        """Close the filter customization modal and validate the filter parameters."""

        trig = ctx.triggered_id

        if trig == 'reset-to-default-btn':
            return True, True, True

        if trig == 'cancel-config-btn':
            return False, True, True

        lowcut_empty = True if not hide_lowcut and lowcut is None else False
        highcut_empty = True if not hide_highcut and highcut is None else False
        order_empty = True if not hide_order and order is None else False
        rp_empty = True if not hide_rp and rp is None else False
        rs_empty = True if not hide_rs and rs is None else False
        window_len_empty = True if not hide_window_len and window_len is None else False
        filter_length_empty = True if not hide_filter_length and filter_length is None else False
        window_type_empty = True if not hide_window_type and window_type is None else False

        hide_empty_param_error = not any([lowcut_empty, highcut_empty, order_empty, rp_empty, rs_empty, window_len_empty, filter_length_empty, window_type_empty])
        if hide_empty_param_error:
            hide_lowcut_highcut_error = False if not hide_lowcut and not hide_highcut and lowcut >= highcut else True
        else:
            hide_lowcut_highcut_error = True

        open_modal = False if hide_empty_param_error and hide_lowcut_highcut_error else True

        return [open_modal, hide_empty_param_error, hide_lowcut_highcut_error]

    # === Validate event timestamps file if provided =========================
    @app.callback(
        [Output('event-load', 'data'),
         Output('event-file-check', 'children', allow_duplicate = True),
         Output('event-uploader', 'children')],
        Input('event-uploader', 'contents'),
        State('event-uploader', 'filename'),
        State('memory-load', 'data'),
        prevent_initial_call = True
    )
    def db_get_event_timestamps(contents, filename, load_data):
        """Read and store event timestamps to memory."""
        if not contents:
            raise PreventUpdate

        file_check = []
        uploaded_file_type = load_data['source']
        filepath = load_data['filename']

        # Check for valid file extensions
        if not _core.io._validate_event_file_ext(filename):
            file_check = [html.I(className = 'fa-solid fa-circle-xmark'),
                          html.Span('Invalid file extension.')]
            uploaded = html.Span('Select File...')
            return None, file_check, uploaded

        event_data = _core.io._parse_event_data(contents, filename)
        is_batch = isinstance(event_data, dict)

        # Validate that the uploaded event files match the batch data files
        if uploaded_file_type == 'batch':
            batch_file = Path(filepath)
            session_path = batch_file.parent
            batch_dir = session_path / 'batch'
            batch = sorted([
                f for f in batch_dir.iterdir()
                if f.is_file() and not f.name.startswith('.') and
                   f.suffix == '.csv'])

            # Batch data requires a Zip of event files keyed by filename
            if not is_batch:
                file_check = [
                    html.I(className = 'fa-solid fa-circle-xmark'),
                    html.Span('Upload a Zip of event files for batch data.')]
                uploaded = html.Span('Select File...')
                return None, file_check, uploaded

            batch_stems = {f.stem for f in batch}
            event_stems = set(event_data.keys())
            missing_event_files = batch_stems - event_stems
            extra_event_files = event_stems - batch_stems
            if missing_event_files or extra_event_files:
                file_check = [
                    html.I(className = 'fa-solid fa-circle-xmark'),
                    html.Span('Event filenames do not match the batch data files.')]
                uploaded = html.Span('Select File...')
                return None, file_check, uploaded

        if not is_batch:
            event_data = {'single': event_data}

        required_event_cols = ['event', 'start', 'end']
        validated = {}
        for key, df in event_data.items():
            df.columns = df.columns.str.lower().str.strip()

            # Check required headers in event file
            if not all(col in df.columns for col in required_event_cols) \
                    or df.empty:
                file_check = [html.I(className = 'fa-solid fa-circle-xmark'),
                              html.Span('Invalid event file contents!')]
                uploaded = html.Span('Select File...')
                return None, file_check, uploaded

            # Check for duplicate event names
            if df['event'].duplicated().any():
                file_check = [html.I(className = 'fa-solid fa-circle-xmark'),
                              html.Span('Duplicate event names found.')]
                uploaded = html.Span('Select File...')
                return None, file_check, uploaded

            # Convert start and end to datetime
            try:
                for col in ['start', 'end']:
                    df[col] = _core.io._convert_timestamps(df[col])
            except Exception as e:
                file_check = [
                    html.I(className = 'fa-solid fa-circle-xmark'),
                    html.Span('Invalid timestamp format.')
                ]
                uploaded = html.Span('Select File...')
                return None, file_check, uploaded

            validated[key] = df

        if filename:
            uploaded = html.Span(f'{filename}')
        else:
            uploaded = html.Span('Select File...')

        # Store as dict of records for a batch, flat records for a single file
        if is_batch:
            store = {k: v.to_dict('records') for k, v in validated.items()}
        else:
            store = validated['single'].to_dict('records')

        return store, file_check, uploaded

    # === Read temperature data file if provided ==============================
    @app.callback(
        [Output('temperature-load', 'data'),
         Output('temp-file-check', 'children'),
         Output('temp-uploader', 'children', allow_duplicate = True)],
        Input('temp-uploader', 'contents'),
        State('temp-uploader', 'filename'),
        prevent_initial_call = True
    )
    def db_get_temperature_data(contents, filename):
        """"Read and store any temperature data to memory."""
        if not contents:
            raise PreventUpdate

        file_check = []
        data = {}

        temperature_data = _core.io._parse_temp_csv(contents)
        if temperature_data.shape[1] != 1:
            file_check = [html.I(className = 'fa-solid fa-circle-xmark'),
                          html.Span('Invalid data type!')]

        col = temperature_data.iloc[:, 0]
        if pd.api.types.is_string_dtype(col) and not col.str.replace(
                '.', '').str.isnumeric().all():
            col = col.iloc[1:]
        temp_vals = pd.to_numeric(col, errors = 'coerce').dropna().tolist()
        data['Temp'] = temp_vals

        # Update uploader text to show the filename
        if filename:
            uploaded = html.Span(f'{filename}')
        else:
            uploaded = html.Span('Select File...')

        return data, file_check, uploaded

    # === Clear uploaded event file ===========================================
    @app.callback(
        [Output('event-uploader', 'contents'),
         Output('event-uploader', 'filename'),
         Output('event-uploader', 'last_modified'),
         Output('event-uploader', 'children', allow_duplicate = True),
         Output('event-file-check', 'children', allow_duplicate = True)],
        Input('clear-event-upload', 'n_clicks'),
        prevent_initial_call = True
    )
    def clear_uploaded_events(n):
        """Reset the event upload component's contents if the 'erase' icon is
        clicked."""
        if n:
            return None, None, None, 'Select File...', []

    # === Clear uploaded temperature file =====================================
    @app.callback(
        [Output('temp-uploader', 'contents'),
         Output('temp-uploader', 'filename'),
         Output('temp-uploader', 'last_modified'),
         Output('temp-uploader', 'children', allow_duplicate = True),
         Output('temp-file-check', 'children', allow_duplicate = True)],
        [Input('clear-temp-upload', 'n_clicks'),
         Input('duplicate-temp-error-modal', 'is_open')],
        prevent_initial_call = True
    )
    def clear_uploaded_temp(n, error_is_open):
        """Reset the temperature upload component's contents if the 'erase'
        icon is clicked or the duplicate temperature input error modal is
        closed."""
        trig = ctx.triggered_id
        if trig == 'clear-temp-upload':
            return None, None, None, 'Select File...', []
        if trig == 'duplicate-temp-error-modal' and not error_is_open:
            return None, None, None, 'Select File...', []
        raise PreventUpdate

    # =================== POPULATE PARAMETERIZATION FIELDS ====================
    @app.callback(
        [Output('setup-data-header', 'hidden'),
         Output('setup-data', 'hidden'),
         Output('preprocess-data', 'hidden', allow_duplicate = True),
         Output('eda-preprocessing', 'hidden', allow_duplicate = True),
         Output('select-scr-detector', 'hidden', allow_duplicate = True),
         Output('segment-data', 'hidden'),
         Output('data-type-container', 'hidden'),     # data types div
         Output('data-types', 'value'),
         Output('data-variables', 'hidden'),          # dropdowns div
         Output('variable-mapping-check', 'hidden'),
         Output('data-type-dropdown-1', 'options'),
         Output('data-type-dropdown-1', 'value'),
         Output('data-type-dropdown-2', 'options'),
         Output('data-type-dropdown-2', 'value'),
         Output('data-type-dropdown-3', 'options'),
         Output('data-type-dropdown-3', 'value'),
         Output('data-type-dropdown-4', 'options'),
         Output('data-type-dropdown-4', 'value'),
         Output('data-type-dropdown-5', 'options'),
         Output('data-type-dropdown-5', 'value'),
         Output('toggle-temp-data', 'on'),
         Output('temp-variable', 'options'),
         Output('temp-variable', 'value'),
         Output('temp-uploader', 'disabled'),
         Output('temp-uploader', 'children', allow_duplicate = True),
         Output('sampling-rate', 'value', allow_duplicate = True),
         Output('by-event-help', 'style'),
         Output('seg-size', 'value', allow_duplicate = True),
         Output('beat-detectors', 'options', allow_duplicate = True),
         Output('beat-detectors', 'value', allow_duplicate = True),
         Output('artifact-method', 'value'),
         Output('artifact-tol', 'value'),
         Output('toggle-filter', 'on'),
         Output('scr-detectors', 'value', allow_duplicate = True),
         Output('eda-valid-min', 'value'),
         Output('eda-valid-max', 'value')],
        [Input('memory-load', 'data'),
         Input('config-memory', 'data'),
         State('toggle-config', 'on')],
        prevent_initial_call = True
    )
    def db_handle_upload_params(memory, configs, toggle_config_on):
        """Output parameterization fields according to uploaded data."""
        loaded = ctx.triggered_id
        if loaded is None or memory == 'invalid':
            raise PreventUpdate

        # Default visibility
        hide_setup_header = False
        hide_setup = False
        hide_preprocess = True
        hide_eda_preprocess = True
        hide_segment_data = False
        hide_data_types = False
        hide_data_vars = False
        hide_variable_error = True

        # Default toggler states
        temp_on = False
        filter_on = True

        # Default parameter values
        base_headers = ['<Var>', '<Var>']
        drop_values = [base_headers[:] for _ in range(6)]
        temp_uploader_disabled = False
        temp_uploader_text = 'Select File...'
        temp_options = []
        temp_value = None
        artifact_method = 'cbd'
        artifact_tol = 1
        scr_detector = 'threshold'
        seg_size = 60
        fs = 500
        dtype = None
        eda_min = 0.2
        eda_max = 40

        by_event_help = {}
        beat_detectors = []
        default_beat_detector = None

        if loaded == 'memory-load':

            # -- device sources ----------------------------------------------
            if memory['source'] == 'Actiwave':
                hide_setup = True
                hide_data_types = True
                hide_data_vars = True
                hide_preprocess = False
                dtype = 'ECG'
                beat_detectors = [
                    {'label': 'Manikandan & Soman (2012)',
                     'value': 'manikandan'},
                    {'label': 'Engels & Zeelenberg (1979)', 'value': 'engzee'},
                    {'label': 'Nabian et al. (2018)', 'value': 'nabian'},
                    {'label': 'Pan & Tompkins (1985)', 'value': 'pantompkins'}
                ]
                default_beat_detector = 'manikandan'
                if toggle_config_on:
                    seg_size = configs['segment size']
                    fs = configs['sampling rate']
                    dtype = configs['data type']

            elif memory['source'] == 'E4':
                hide_setup = True
                hide_eda_preprocess = False
                hide_data_types = True
                hide_data_vars = True
                fs = 64
                seg_size = 180
                if toggle_config_on:
                    seg_size = configs['segment size']
                    fs = configs['sampling rate']
                    dtype = configs['data type']
                    scr_detector = configs['scr detector']

            # -- csv sources -------------------------------------------------
            elif memory['source'] == 'csv':
                if toggle_config_on:
                    pass
                else:
                    headers = _core.io._get_csv_headers(memory['filename'])
                    base_headers = headers
                    drop_values = [headers[:] for _ in range(6)]

            # -- batch sources -----------------------------------------------
            elif memory['source'] == 'batch':
                if toggle_config_on:
                    pass
                else:
                    session_path = Path(memory['filename']).parent
                    extract_dir = session_path / 'batch'
                    with zipfile.ZipFile(memory['filename'], 'r') as zf:
                        batch_headers = []

                        # Filter out macOS metadata in zip file
                        zfiles = [f for f in zf.namelist()
                                  if f.lower().endswith('.csv')
                                  and not f.endswith('/')
                                  and not f.startswith('__MACOSX/')
                                  and '/._' not in f
                                  and not f.endswith('.DS_Store')]

                        for f in zfiles:
                            fname = Path(f).name
                            if _core.io._check_csv(f):
                                extracted_path = zf.extract(
                                    f, path = str(extract_dir))

                                # Move to root 'temp' directory
                                root_temp = Path(extract_dir) / fname
                                shutil.move(extracted_path, root_temp)

                                # Get CSV headers
                                hdrs = _core.io._get_csv_headers(str(root_temp))
                                batch_headers.append(tuple(hdrs))

                        # Clean up unnecessary directories
                        for item in extract_dir.iterdir():
                            if item.is_dir():
                                shutil.rmtree(item, ignore_errors = True)

                    # Check if any headers differ across files
                    unique = set(batch_headers)
                    if len(unique) > 1:
                        hide_variable_error = False
                    elif len({tuple(h) for h in batch_headers}) == 1:
                        headers = list(unique.pop())
                        base_headers = headers
                        drop_values = [headers[:] for _ in range(6)]

                # Disable temperature data upload component
                temp_uploader_disabled = True
                temp_uploader_text = 'Enabled for single-file uploads only.'

        elif loaded == 'config-memory':
            device = configs['source']
            dtype = configs['data type']
            seg_size = configs['segment size']
            fs = configs['sampling rate']
            artifact_method = configs['artifact identification method']
            artifact_tol = configs['artifact tolerance']
            filter_on = configs['filters']
            scr_detector = configs['scr detector']
            temp_on = configs['use temperature']
            eda_min = configs['minimum eda']
            eda_max = configs['maximum eda']

            if device in ('E4', 'Actiwave'):
                hide_setup = hide_data_types = hide_data_vars = True
                base_headers = []
                drop_values = [[] for _ in range(6)]
            else:
                headers = list(configs['headers'].values())
                base_headers = headers
                drop_values = [[h for h in configs['headers']
                                if h is not None] for _ in range(6)]

            # Populate temperature dropdown
            if temp_on:
                tv = configs.get('temperature variable')
                if tv:
                    temp_options = [{'label': tv, 'value': tv}]
                    temp_value = tv

        dropdown_options = [{'label': h, 'value': h} for h in base_headers
                            if h is not None]

        if not temp_options:
            temp_options = dropdown_options[:]
        if temp_value is not None:
            pass
        else:
            temp_value = None

        return (
            hide_setup_header, hide_setup, hide_preprocess,
            hide_eda_preprocess,   # eda-preprocessing
            hide_eda_preprocess,   # select-scr-detector
            hide_segment_data, hide_data_types, dtype,
            hide_data_vars, hide_variable_error,

            # variable dropdowns
            dropdown_options, drop_values[0],
            dropdown_options, drop_values[1],
            dropdown_options, drop_values[2],
            dropdown_options, drop_values[3],
            dropdown_options, drop_values[4],

            # temperature toggle and dropdown options
            temp_on, temp_options, temp_value,

            # temperature data upload
            temp_uploader_disabled, temp_uploader_text,

            fs, by_event_help, seg_size,
            beat_detectors, default_beat_detector, artifact_method,
            artifact_tol, filter_on, scr_detector, eda_min, eda_max
        )

    # =================== TOGGLE EXPORT CONFIGURATION MODAL ===================
    @app.callback(
        [Output('config-download-memory', 'clear_data'),
         Output('config-modal', 'is_open'),
         Output('config-description', 'hidden'),
         Output('config-check', 'hidden'),
         Output('config-modal-btns', 'hidden'),
         Output('config-close-btn', 'hidden')],
        [Input('configure', 'n_clicks'),
         Input('close-config1', 'n_clicks'),
         Input('close-config2', 'n_clicks'),
         Input('config-download-memory', 'data'),
         State('config-modal', 'is_open')],
        prevent_initial_call = True
    )
    def toggle_config_modal(n, n1, n2, config_data, is_open):
        """Open and close the Export Configuration modal."""
        hide_config_desc = False  # show export fields
        hide_config_check = True
        hide_config_btns = False  # show 'configure' and 'cancel'
        hide_config_close = True

        if is_open is True:
            # If 'Cancel' or 'Done' is clicked
            if n1 or n2:
                # Reset the content and close the modal
                return [True, not is_open,
                        hide_config_desc, hide_config_check,
                        hide_config_btns, hide_config_close]

            # If a configuration file was created and exported
            hide_config_desc = True
            hide_config_check = False
            hide_config_btns = True
            hide_config_close = False
            if config_data is not None:
                # Keep the modal open and show export confirmation
                return [True, is_open,
                        hide_config_desc, hide_config_check,
                        hide_config_btns, hide_config_close]
            else:
                return [False, is_open,
                        hide_config_desc, hide_config_check,
                        hide_config_btns, hide_config_close]

        else:
            # If 'Save' is clicked
            if ctx.triggered_id == 'configure':
                if config_data is not None:
                    return [True, not is_open,
                            hide_config_desc, hide_config_check,
                            hide_config_btns, hide_config_close]
                else:
                    return [False, not is_open,
                            hide_config_desc, hide_config_check,
                            hide_config_btns, hide_config_close]

        return [False, is_open, hide_config_desc, hide_config_check,
                hide_config_btns, hide_config_close]

    # ====================== CREATE AND SAVE CONFIG FILE ======================
    @app.callback(
        [Output('config-file-download', 'data'),
         Output('config-download-memory', 'data')],
        [Input('config-btn', 'n_clicks'),
         State('memory-load', 'data'),
         State('data-types', 'value'),
         State('sampling-rate', 'value'),
         State('data-type-dropdown-1', 'value'),
         State('data-type-dropdown-2', 'value'),
         State('data-type-dropdown-3', 'value'),
         State('data-type-dropdown-4', 'value'),
         State('data-type-dropdown-5', 'value'),
         State('seg-size', 'value'),
         State('artifact-method', 'value'),
         State('artifact-tol', 'value'),
         State('toggle-filter', 'on'),
         State('toggle-temp-data', 'on'),
         State('temp-variable', 'value'),
         State('scr-detectors', 'value'),
         State('scr-amp-thresh', 'value'),
         State('eda-valid-min', 'value'),
         State('eda-valid-max', 'value'),
         State('config-filename', 'value')],
        prevent_initial_call = True
    )
    def write_confirm_config(n, data, dtype, fs, d1, d2, d3, d4, d5,
                             seg_size, artifact_method, artifact_tol,
                             filter_on, temp_on, temp_var, scr_detector,
                             min_peak_amp, eda_min, eda_max, filename):
        """Export the configuration file."""
        if n:
            headers = None
            device = data['source'] if data['source'] != 'csv' else 'Other'
            if device == 'Actiwave':
                actiwave = physioview.Actiwave(data['filename'])
                fs = actiwave.get_ecg_fs()
                dtype = 'ECG'
            elif device == 'E4':
                E4 = physioview.Empatica(data['filename'])
                fs = E4.get_bvp().fs
                dtype = 'BVP'
            else:
                headers = {
                    'Time/Sample': d1,
                    'Signal': d2,
                    'X': d3,
                    'Y': d4,
                    'Z': d5}
            json_object = _core.io._create_configs(
                device, dtype, fs, seg_size, artifact_method, artifact_tol,
                filter_on, scr_detector, min_peak_amp, headers, temp_on,
                temp_var, eda_min, eda_max)
            download = {'content': json_object, 'filename': f'{filename}.json'}
            return [download, 1]

    # ============================= RUN PIPELINE ==============================
    @callback(
        output = [
            Output('dtype-validator', 'is_open', allow_duplicate = True),
            Output('mapping-validator', 'is_open'),
            Output('pipeline-error-modal', 'is_open'),
            Output('event-file-error-modal', 'is_open'),
            Output('duplicate-temp-error-modal', 'is_open'),
            Output('memory-db', 'data'),
        ],
        inputs = [
            Input('run-data', 'n_clicks'),
            State('memory-load', 'data'),
            State('e4-data-types', 'value'),
            State('data-types', 'value'),
            State('sampling-rate', 'value'),
            State('resampling-rate', 'value'),
            State('data-type-dropdown-1', 'value'),
            State('data-type-dropdown-2', 'value'),
            State('data-type-dropdown-3', 'value'),
            State('data-type-dropdown-4', 'value'),
            State('data-type-dropdown-5', 'value'),
            State('toggle-event-segmentation', 'on'),
            State('event-load', 'data'),
            State('temperature-load', 'data'),
            State('temp-variable', 'value'),
            State('beat-detectors', 'value'),
            State('peak-detection-mode', 'value'),
            State('seg-size', 'value'),
            State('artifact-method', 'value'),
            State('artifact-tol', 'value'),
            State('toggle-filter', 'on'),
            State('filter-lowcut', 'value'),
            State('filter-highcut', 'value'),
            State('filter-order', 'value'),
            State('filter-rp', 'value'),
            State('filter-rs', 'value'),
            State('filter-window-len', 'value'),
            State('filter-length', 'value'),
            State('filter-window-type', 'value'),
            State('scr-detectors', 'value'),
            State('scr-amp-thresh', 'value'),
            State('eda-valid-min', 'value'),
            State('eda-valid-max', 'value'),
        ],
        background = True,
        running = [
            (Output('progress-bar', 'style'),
             {'visibility': 'visible'}, {'visibility': 'hidden'}),
            (Output('stop-run', 'hidden'), False, True),
            (Output('run-data', 'disabled'), True, False),
            (Output('configure', 'disabled'), True, False)
        ],
        cancel = [Input('stop-run', 'n_clicks')],
        progress = [
            Output('progress-bar', 'value'),
            Output('progress-bar', 'label')
        ],
        prevent_initial_call = True
    )
    def run_pipeline(set_progress, n, load_data, e4_dtype, dtype, fs, rs,
                     d1, d2, d3, d4, d5, event_toggle_on, event_times,
                     temp_data, temp_var, beat_detector, beat_detection_mode,
                     seg_size, artifact_method, artifact_tol, filter_on,
                     filter_lowcut, filter_highcut, filter_order,
                     filter_rp, filter_rs, filter_window_len, filter_len,
                     filter_window_type, scr_detector, min_peak_amp,
                     eda_min, eda_max):
        """Read Actiwave Cardio, Empatica E4, or CSV-formatted data, save
        the data to the local memory, and load the progress spinner."""

        dtype_error = False
        map_error = False
        pipeline_error = False
        event_file_error = False
        temp_input_error = False
        _errors = lambda: (dtype_error, map_error, pipeline_error,
                           event_file_error, temp_input_error, None)

        # Set up storage
        memory = {}

        if ctx.triggered_id == 'run-data':

            # Reset progress bar
            set_progress((0, '0%'))

            # Create '_render' folder
            if render_dir.exists():
                shutil.rmtree(render_dir)
            render_dir.mkdir(parents = True, exist_ok = True)

            file_type = load_data['source']
            if file_type not in ('Actiwave', 'E4'):
                if dtype is None:
                    dtype_error = True
                    return _errors()
                elif d2 is None:
                    map_error = True
                    return _errors()
            else:
                if file_type == 'E4':
                    dtype = 'EDA' if e4_dtype == 'EDA' else 'PPG'
                elif file_type == 'Actiwave':
                    dtype = 'ECG'

            filepath = load_data['filename']
            filename = Path(filepath).name  # e.g., "example.csv"
            file = Path(filepath)

            # Check for event data
            if event_toggle_on:
                if event_times is None or len(event_times) == 0:
                    event_file_error = True
                    return _errors()
            memory['segment by event'] = event_toggle_on

            # Enable downsampling if fs or rs is greater than the sampling
            # rate (~250 Hz) of the render data
            ds = fs > 250 and (rs is None or rs > 250)
            ds_data, ds_ibi, ds_acc, ds_fs = None, None, None, None

            # Initialize for uploads without IBI or ACC
            ibi, acc = None, None
            event_durations = {}

            # Get peak detector according to signal type
            if dtype in ('ECG', 'PPG'):
                peak_detector = beat_detector
            else:
                peak_detector = scr_detector

            # Initialize data preprocessing class
            filter_kwargs = {
                k: v for k, v in {
                    'lowcut': filter_lowcut,
                    'highcut': filter_highcut,
                    'order': filter_order,
                    'rp': filter_rp,
                    'rs': filter_rs,
                    'window_len': filter_window_len,
                    'filter_length': filter_len,
                    'window_type': filter_window_type,
            }.items() if v is not None}

            # -- batch sources -----------------------------------------------
            if file_type == 'batch':
                batch_file = Path(filepath)
                session_path = batch_file.parent
                batch_dir = session_path / 'batch'
                batch = sorted([
                    f for f in batch_dir.iterdir()
                    if f.is_file() and not f.name.startswith('.') and
                       f.suffix == '.csv'])

                # Set progress bar total
                total_progress = len(batch) + 1
                perc = (1 / total_progress) * 100
                set_progress((perc, f'{perc:.0f}%'))
                sleep(0.5)

                # Preprocess each file in the batch
                for idx, f in enumerate(batch):
                    fname = f.stem

                    # Get event times for each file
                    if event_toggle_on and isinstance(event_times, dict):
                        event_df = pd.DataFrame(event_times[fname])
                        event_df['start'] = pd.to_datetime(event_df['start'])
                        event_df['end'] = pd.to_datetime(event_df['end'])
                    else:
                        event_df = None

                    # Initialize data preprocessing object for each file
                    preprocessor = _core.Preprocessor(
                        dtype, fs, filter_on, peak_detector,
                        beat_detection_mode, event_df,
                        seg_size, filter_kwargs)

                    # If timestamps are given
                    if d1 is not None:
                        has_ts = True
                        # No acceleration data
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data, acc = _core.io._setup_data(
                                f, dtype, [d1, d2], temp_var, event_df,
                                has_ts)
                        # With acceleration data
                        else:
                            data, acc = _core.io._setup_data(
                                f, dtype, [d1, d2, d3, d4, d5], temp_var,
                                event_df, has_ts)
                    else:
                        has_ts = False
                        # No acceleration data
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data, acc = _core.io._setup_data(
                                f, dtype, [d2], temp_var, event_df, has_ts)
                        # With acceleration data
                        else:
                            data, acc = _core.io._setup_data(
                                f, dtype, [d2, d3, d4, d5], temp_var,
                                event_df, has_ts)

                    # Preprocess any acceleration data
                    if acc is not None:
                        acc['Magnitude'] = ACC.compute_magnitude(
                            acc['X'], acc['Y'], acc['Z'])
                        if has_ts:
                            unix_fmt = _core.io._check_unix(acc.Timestamp)
                            if unix_fmt is not None:
                                acc.Timestamp = pd.to_datetime(
                                    acc.Timestamp, unit = unix_fmt)
                        acc.to_csv(
                            str(temp_path / f'{fname}_ACC.csv'),
                            index = False)

                    # ---- cardiac data --------------------------------------
                    if dtype in ('ECG', 'PPG'):

                        # Event-based cardiac batch preprocessing
                        if event_toggle_on:
                            try:
                                durations, ds_fs = _preprocess_cardiac_by_event(
                                    preprocessor, data, fs, dtype, fname,
                                    acc = acc,
                                    ts_col = 'Timestamp' if has_ts else None,
                                    artifact_method = artifact_method,
                                    artifact_tol = artifact_tol)
                                event_durations.update(durations)
                            except RuntimeError:
                                pipeline_error = True
                                return _errors()

                        else:
                            # Segment-based cardiac batch preprocessing
                            try:
                                preprocessed, metrics = preprocessor.preprocess_full(
                                    data, artifact_method = artifact_method,
                                    artifact_tol = artifact_tol)

                                # Check for detected beats
                                beats_ix = preprocessor.peaks_ix
                                if len(beats_ix) == 0:
                                    pipeline_error = True
                                    return _errors()

                                # Downsample preprocessed data for rendering
                                artifacts_ix = preprocessor.artifacts_ix
                                ds_data, ds_ibi, _, ds_acc, ds_fs = \
                                    _core.io._downsample_data(
                                        preprocessed, fs, dtype, beats_ix,
                                        artifacts_ix, acc = acc)

                            except Exception as e:
                                pipeline_error = True
                                print(traceback.format_exc())
                                return _errors()

                            # Write IBI data to 'temp' folder
                            ibi = physioview.compute_ibis(
                                data, fs, beats_ix,
                                ts_col = 'Timestamp' if has_ts else None)
                            ibi.to_csv(
                                str(temp_path / f'{fname}_IBI.csv'), index = False)

                    # ---- EDA data ------------------------------------------
                    else:
                        temp = data['Temp'].values if 'Temp' in data.columns \
                            else None

                        # Event-based EDA batch preprocessing
                        if event_toggle_on:
                            try:
                                durations, ds_fs = _preprocess_eda_by_event(
                                    preprocessor, data, fname, acc = acc,
                                    rs = rs, min_peak_amp = min_peak_amp,
                                    temp = temp, eda_min = eda_min,
                                    eda_max = eda_max)
                                event_durations.update(durations)
                            except RuntimeError:
                                pipeline_error = True
                                return _errors()

                        # Segment-based EDA batch preprocessing
                        else:
                            try:
                                preprocessed, metrics = preprocessor.preprocess_full(
                                    data, rs, min_peak_amp, temp_data = temp,
                                    eda_min = eda_min, eda_max = eda_max)
                            except Exception:
                                pipeline_error = True
                                print(traceback.format_exc())
                                return _errors()

                            # Downsample data for rendering
                            ds_data, ds_ibi, _, ds_acc, ds_fs = \
                                _core.io._downsample_data(
                                    preprocessed, preprocessor.fs, dtype,
                                    preprocessor.peaks_ix,
                                    preprocessor.artifacts_ix, acc = acc)

                    if not event_toggle_on:
                        # Write preprocessed data and metrics to 'temp' folder
                        preprocessed.to_csv(
                            str(temp_path / f'{fname}_{dtype}.csv'), index = False)
                        metrics.to_csv(
                            str(temp_path / f'{fname}_SQA.csv'), index = False)

                        # Write any downsampled data to '_render' folder
                        _core.io._create_render(fname, ds_data, ds_ibi, ds_acc)

                    # Update progress bar
                    perc = ((idx + 2) / total_progress) * 100
                    set_progress((perc, f'{perc:.0f}%'))
                    sleep(0.5)

            # -- single-file sources -----------------------------------------
            else:
                # Update progress bar for all single-file sources: 33%
                total_progress = 6
                perc = (2 / total_progress) * 100
                set_progress((perc, f'{perc:.0f}%'))
                sleep(0.5)

                ts_col = None

                # Get event times for the single file
                if event_toggle_on and event_times is not None \
                        and not isinstance(event_times, dict):
                    event_df = pd.DataFrame(event_times)
                    event_df['start'] = pd.to_datetime(event_df['start'])
                    event_df['end'] = pd.to_datetime(event_df['end'])
                else:
                    event_df = None

                # Initialize data preprocessing object
                preprocessor = _core.Preprocessor(
                    dtype, fs, filter_on, peak_detector, beat_detection_mode,
                    event_df, seg_size, filter_kwargs)

                if file_type in ('Actiwave', 'E4'):

                    # -- Actiwave Cardio sources -----------------------------
                    if file_type == 'Actiwave':
                        dtype = 'ECG'

                        # Prepare Actiwave Cardio data
                        actiwave = physioview.Actiwave(filepath)
                        actiwave_data = actiwave.preprocess(time_aligned = True)
                        data = actiwave_data[['Timestamp', dtype]].copy()
                        acc = actiwave_data[['Timestamp', 'X', 'Y', 'Z']].copy()
                        acc.to_csv(
                            str(temp_path / f'{file.stem}_ACC.csv'), index = False)
                        fs = actiwave.get_ecg_fs()
                        ts_col = 'Timestamp'

                    # -- Empatica E4 sources ---------------------------------
                    elif file_type == 'E4':
                        E4 = physioview.Empatica(filepath)
                        e4_data = E4.preprocess()

                        # Accelerometer data
                        acc = e4_data.acc
                        acc.to_csv(
                            str(temp_path / f'{file.stem}_ACC.csv'), index = False)

                        # Extract and save EDA data
                        if e4_dtype == 'EDA':
                            dtype = 'EDA'
                            eda = e4_data.eda
                            eda.to_csv(
                                str(temp_path / f'{file.stem}_EDA.csv'), index = False)
                            fs = e4_data.eda_fs
                            data = eda.copy()

                            # Extract accompanying skin temperature data
                            temp = e4_data.temp
                            temp.rename(columns = {'TEMP': 'Temp'}, inplace = True)
                            temp.to_csv(
                                str(temp_path / f'{file.stem}_TEMP.csv'), index = False)

                        # Extract and save BVP data
                        elif e4_dtype == 'PPG':
                            bvp = e4_data.bvp
                            bvp.to_csv(
                                str(temp_path / f'{file.stem}_BVP.csv'), index = False)
                            fs = e4_data.bvp_fs
                            data = bvp.copy()

                        ts_col = 'Timestamp'

                    # Reset preprocessor with device-specific sampling rates
                    preprocessor = _core.Preprocessor(
                        dtype, fs, filter_on, peak_detector, beat_detection_mode,
                        event_df, seg_size, filter_kwargs)

                # -- csv sources ---------------------------------------------
                else:

                    # Check if duplicate temperature inputs
                    if temp_data is not None and temp_var is not None:
                        temp_input_error = True
                        return dtype_error, map_error, pipeline_error, \
                            event_file_error, temp_input_error, None

                    # If timestamps are given
                    if d1 is not None:
                        has_ts = True
                        ts_col = 'Timestamp'
                        # No acceleration data
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data, acc = _core.io._setup_data(
                                filepath, dtype, [d1, d2], temp_var,
                                event_df, has_ts)
                        # With acceleration data
                        else:
                            data, acc = _core.io._setup_data(
                                filepath, dtype, [d1, d2, d3, d4, d5],
                                temp_var, event_df, has_ts)
                    else:
                        has_ts = False
                        # No acceleration data
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data, acc = _core.io._setup_data(
                                filepath, dtype, [d2], temp_var,
                                event_df, has_ts)
                        # With acceleration data
                        else:
                            data, acc = _core.io._setup_data(
                                filepath, dtype, [d2, d3, d4, d5], temp_var,
                                event_df, has_ts)

                # Update progress bar: 50%
                perc = (3 / total_progress) * 100
                set_progress((perc, f'{perc:.0f}%'))
                sleep(0.5)

                # Preprocess any acceleration data
                if acc is not None and not acc.empty:
                    acc['Magnitude'] = ACC.compute_magnitude(
                        acc['X'], acc['Y'], acc['Z'])
                    if 'Timestamp' in acc.columns:
                        unix_fmt = _core.io._check_unix(acc.Timestamp)
                        if unix_fmt is not None:
                            acc.Timestamp = pd.to_datetime(
                                acc.Timestamp, unit = unix_fmt)
                    acc.to_csv(str(temp_path / f'{file.stem}_ACC.csv'),
                               index = False)

                # Update progress bar: 67%
                perc = (4 / total_progress) * 100
                set_progress((perc, f'{perc:.0f}%'))
                sleep(0.5)

                # Preprocess any cardiac data
                if dtype in ('ECG', 'PPG') or e4_dtype == 'PPG':

                    # Event-based cardiac preprocessing
                    # if segment_by_event:
                    if event_toggle_on:
                        try:
                            durations, ds_fs = _preprocess_cardiac_by_event(
                                preprocessor, data, fs, dtype, file.stem,
                                acc = acc, ts_col = ts_col,
                                artifact_method = artifact_method,
                                artifact_tol = artifact_tol)
                            event_durations.update(durations)
                        except RuntimeError:
                            pipeline_error = True
                            return _errors()

                    # Segment-based cardiac preprocessing
                    else:
                        try:
                            preprocessed, metrics = preprocessor.preprocess_full(
                                data, artifact_method = artifact_method,
                                artifact_tol = artifact_tol)
                        except Exception as e:
                            pipeline_error = True
                            print(traceback.format_exc())
                            return dtype_error, map_error, pipeline_error, \
                                event_file_error, temp_input_error, None

                        # Compute and write IBI data to 'temp' folder
                        beats_ix = preprocessor.peaks_ix
                        ibi = physioview.compute_ibis(
                            preprocessed, fs, beats_ix, ts_col)
                        ibi.to_csv(temp_path / f'{file}_IBI.csv', index = False)

                        # Downsample data for rendering
                        artifacts_ix = preprocessor.artifacts_ix
                        ds_data, ds_ibi, _, ds_acc, ds_fs = \
                            _core.io._downsample_data(
                                preprocessed, fs, dtype, beats_ix,
                                artifacts_ix, acc = acc)

                        # Write SQA metrics to 'temp' folder
                        metrics.to_csv(temp_path / f'{file.stem}_SQA.csv', index = False)

                # Preprocess any EDA data
                if dtype == 'EDA' or e4_dtype == 'EDA':
                    if temp_data is not None:
                        temp = temp_data['Temp']
                    elif 'Temp' in data.columns:
                        temp = data['Temp'].values
                    elif (temp_path / f'{file}_TEMP.csv').exists():
                        temperature = pd.read_csv(str(temp_path / f'{file}_TEMP.csv'))
                        temperature.Timestamp = pd.to_datetime(temperature.Timestamp)
                        data = pd.merge(data, temperature, on = 'Timestamp',
                                        how = 'inner')
                        temp = data['Temp'].values
                    else:
                        temp = None

                    # Event-based preprocessing
                    if event_toggle_on:
                        try:
                            durations, ds_fs = _preprocess_eda_by_event(
                                preprocessor, data, file.stem, acc = acc,
                                rs = rs, min_peak_amp = min_peak_amp,
                                temp = temp, eda_min = eda_min, eda_max = eda_max)
                            event_durations.update(durations)
                        except RuntimeError:
                            pipeline_error = True
                            return _errors()

                    # Segment-based EDA preprocessing
                    else:
                        try:
                            preprocessed, metrics = preprocessor.preprocess_full(
                                data, rs, min_peak_amp, temp_data = temp,
                                eda_min = eda_min, eda_max = eda_max)
                        except Exception as e:
                            pipeline_error = True
                            print(traceback.format_exc())
                            return dtype_error, map_error, pipeline_error, \
                                event_file_error, temp_input_error, None

                        # Downsample EDA data for rendering
                        peaks_ix = preprocessor.peaks_ix
                        artifacts_ix = preprocessor.artifacts_ix
                        ds_data, ds_ibi, _, ds_acc, ds_fs = _core.io._downsample_data(  # ds_ibi = None
                            preprocessed, preprocessor.fs, dtype,
                            peaks_ix, artifacts_ix, acc = acc)

                # Write preprocessed and downsampled data to 'temp/' directory
                if not event_toggle_on:
                    preprocessed.to_csv(
                        str(temp_path / f'{file.stem}_{dtype}.csv'), index = False)
                    metrics.to_csv(
                        str(temp_path / f'{file.stem}_SQA.csv'), index = False)

                    # to '_render' directory
                    _core.io._create_render(file.stem, ds_data, ds_ibi, ds_acc)

                # Update progress bar: 83%
                perc = (5 / total_progress) * 100
                set_progress((perc, f'{perc:.0f}%'))
                sleep(0.5)

            # Store data variables in memory
            memory['file type'] = file_type
            memory['data type'] = dtype
            memory['fs'] = fs
            memory['downsampled fs'] = ds_fs if ds else fs
            memory['filename'] = filename
            memory['duration'] = preprocessor.duration
            memory['event durations'] = event_durations

            # Update progress bar: 100%
            set_progress((100, '100%'))
            sleep(1)

            return [dtype_error, map_error, pipeline_error,
                    event_file_error, temp_input_error, memory]

    # == Recompute SQA metrics for re-rendering ==============================
    @app.callback(
        Output('re-render-sqa-flag', 'data'),
        [Input('beat-correction-status', 'data'),
         Input('be-edited-trigger', 'children')],
        [State('memory-db', 'data'),
         State('data-dropdown', 'value'),
         State('event-dropdown', 'value'),
         State('seg-size', 'value')],
        prevent_initial_call = True
    )
    def recompute_sqa(beat_correction_status, beats_edited, memory,
                      selected_subject, selected_event, segment_size):
        """Recompute signal quality metrics after beat corrections or edits."""
        trig = ctx.triggered_id
        if trig == 'beat-correction-status':
            if selected_subject not in beat_correction_status.keys():
                return False
            elif beat_correction_status[selected_subject] == 'suggested':
                return False
        elif trig == 'be-edited-trigger':
            if beats_edited != selected_subject:
                return False

        fs = memory['fs']
        data_type = memory['data type']
        beat_editor_fs = memory['downsampled fs']
        sqa = SQA.Cardio(fs)
        file = f'{selected_subject}_{selected_event}' if selected_event \
            else selected_subject

        preprocessed_data = pd.read_csv(
            temp_path / f'{file}_{data_type}.csv')

        # Get manual beat edits and recomputed artifacts
        edited_file = temp_path / f'{file}_edited.csv'
        if edited_file.exists():
            edited = pd.read_csv(edited_file)
            edited_beats_ix = edited[edited['Edited Beat'] == 1].index.values
            edited_artifacts_ix = edited[edited['Artifact'] == 1].index.values

            # Map edited indices back to original sampling rate
            beats_ix = _core.beat_editing._map_beat_edits(
                edited_beats_ix, beat_editor_fs, fs)
            artifacts_ix = _core.beat_editing._map_beat_edits(
                edited_artifacts_ix, beat_editor_fs, fs)

            # Remove any existing 'Beat' or 'Artifact' columns to prevent
            # stale values from affecting SQA recomputation
            if 'Beat' in preprocessed_data.columns:
                del preprocessed_data['Beat']
            if 'Artifact' in preprocessed_data.columns:
                del preprocessed_data['Artifact']

        # Get auto-corrected beats and recomputed artifacts
        else:
            beats_ix = preprocessed_data[
                preprocessed_data.Beat == 1].index.values
            artifacts_ix = preprocessed_data[
                preprocessed_data.Artifact == 1].index.values

        ts_col = 'Timestamp' if 'Timestamp' in preprocessed_data.columns else None

        # Set segment size for entire event if segment size is empty
        segment_by_event = memory['segment by event']
        if segment_by_event and segment_size is None:
            segment_size = len(preprocessed_data)

        # Compute SQA metrics
        metrics = sqa.compute_metrics(
            preprocessed_data, beats_ix, artifacts_ix, ts_col,
            seg_size = segment_size, show_progress = False)
        metrics.to_csv(str(temp_path / f'{file}_SQA.csv'), index = False)

        return True

    # == Create Beat Editor editing files =====================================
    @app.callback(
        [Output('beat-editor-spinner', 'children'),
         Output('beat-editor-spinner', 'spinner_class_name'),
         Output('open-beat-editor', 'disabled', allow_duplicate = True)],
        [Input('data-dropdown', 'options'),
         Input('data-dropdown', 'value'),
         Input('event-dropdown', 'options'),
         Input('beat-correction-status', 'data')],
        [State('memory-db', 'data'),
         State('toggle-filter', 'on'),
         State('be-edited-trigger', 'children')],
        prevent_initial_call = True
    )
    def create_beat_editor_files(all_subjects, selected_subject, all_events,
                                 beat_correction_status, memory, filter_on,
                                 prev_beats_edited):
        """Create Beat Editor _edit.json files for uploaded cardiac files and
        enable the 'Beat Editor' button."""
        if memory is None:
            return None, True

        file_type = memory['file type']
        data_type = memory['data type']
        segment_by_event = memory['segment by event']
        trig = ctx.triggered_id

        # Beat Editor button icon
        btn_icon = html.I(className = 'fa-solid fa-arrow-up-right-from-square')

        if data_type not in ('ECG', 'PPG', 'BVP'):
            return btn_icon, '', True

        if prev_beats_edited == selected_subject:
            return btn_icon, 'no-spin', False

        fs = memory['fs']
        signal_col = 'Filtered' if filter_on else data_type

        def _write_beat_editor_file(filename, batch):
            """Read CSV, downsample, and write a beat editor JSON file."""
            data = pd.read_csv(temp_path / f'{filename}_{data_type}.csv')
            ts_col = 'Timestamp' if 'Timestamp' in data.columns else None
            beats_ix = data[data.Beat == 1].index.values
            artifacts_ix = (data[data.Artifact == 1].index.values
                            if 'Artifact' in data.columns else None)

            ds, _, _, _, ds_fs = _core.io._downsample_data(
                data, fs, data_type, beats_ix, artifacts_ix)
            physioview.write_beat_editor_file(
                ds, ds_fs, signal_col, 'Beat', ts_col, filename,
                batch = batch, verbose = False)

        # Build list of (filename, batch) pairs to process
        if file_type == 'batch' and trig != 'beat-correction-status':
            subjects = sorted(all_subjects.values())
            batch = True
        elif file_type == 'batch' or segment_by_event:
            subjects = [selected_subject]
            batch = True
        else:
            subjects = [Path(memory['filename']).stem]
            batch = False

        # Process each subject, splitting by event when applicable
        events = sorted(all_events.values()) if segment_by_event else [None]
        for name in subjects:
            for event in events:
                stem = f'{name}_{event}' if event else name
                if not (temp_path / f'{stem}_{data_type}.csv').exists():
                    continue
                _write_beat_editor_file(stem, batch = batch)

        return btn_icon, '', False

    # ===================== ARTIFACT IDENTIFICATION MODAL =====================
    @app.callback(
        Output('artifact-identification-modal', 'is_open'),
        Input('artifact-method-help', 'n_clicks'),
        prevent_initial_call = True
    )
    def toggle_artifact_identification_help(n):
        if n:
            return True

    # ===================== CONTROL DASHBOARD ELEMENTS ========================
    # === Toggle offcanvas ====================================================
    @app.callback(
        Output('offcanvas', 'is_open', allow_duplicate = True),
        Input('reload-data', 'n_clicks'),
        prevent_initial_call = True
    )
    def reload_data(n):
        """Open and close the offcanvas."""
        if n == 0:
            raise PreventUpdate
        else:
            return True

    # === Populate dropdowns ================================================
    @app.callback(
        [Output('data-dropdown', 'options'),
         Output('data-dropdown', 'value'),
         Output('data-dropdown', 'disabled'),
         Output('event-dropdown', 'options'),
         Output('event-dropdown', 'value'),
         Output('event-dropdown', 'disabled'),
         Output('data-dropdown-icon', 'children'),
         Output('qa-charts-dropdown', 'options'),
         Output('qa-charts-dropdown', 'value')],
        Input('memory-db', 'data'),
        prevent_initial_call = True
    )
    def update_data_select_dropdown(memory):
        """Populate dropdowns with the names of uploaded files and SQA chart
        types according to uploaded data type."""
        file_type = memory['file type']
        data_type = memory['data type']
        segment_by_event = memory['segment by event']
        subject_drop_disabled = True  # dropdown is disabled by default
        event_drop_disabled = True  # dropdown is disabled by default
        event_drop_options = {}
        event_drop_value = None
        sqa_drop_options = []  # empty SQA chart dropdown by default
        sqa_drop_value = ''
        dropdown_icon = html.I(className = 'fa-solid fa-user')

        # Handle batch files
        if file_type == 'batch':
            subject_drop_disabled = False
            filenames = sorted(
                [p.name for p in render_dir.iterdir() if (p.is_dir())])
            if segment_by_event:
                # render dirs are already {subject}_{event}; strip event suffix
                subjects = sorted(set(f.rsplit('_', 1)[0] for f in filenames))
                data_drop_options = {s: s for s in subjects}
                data_drop_value = subjects[0]
            else:
                data_drop_options = {name: name for name in filenames}
                data_drop_value = filenames[0]

        # Handle single E4, Actiwave, and CSV files
        else:
            filename = Path(memory['filename']).stem
            data_drop_value = filename
            data_drop_options = {filename: filename}

        # Populate event name dropdown
        if segment_by_event:
            event_drop_disabled = False
            filenames = sorted(
                [p.name for p in render_dir.iterdir() if (p.is_dir())])
            event_names = sorted(
                [f.split('_')[-1] for f in filenames])
            event_drop_options = {name: name for name in event_names}
            event_drop_value = event_names[0]

        # Set SQA dropdown options for cardiac data
        if data_type in ('ECG', 'PPG', 'BVP'):
            sqa_drop_options = [
                {'label': 'Missing Beats', 'value': 'missing'},
                {'label': 'Artifact Beats', 'value': 'artifact'}
            ]
            sqa_drop_value = 'missing'

        # Set SQA dropdown options for EDA data
        elif data_type == 'EDA':
            sqa_drop_options = [
                {'label': 'Data Validity', 'value': 'validity'},
                {'label': 'Quality Checks', 'value': 'quality'}
            ]
            sqa_drop_value = 'validity'

        return data_drop_options, data_drop_value, subject_drop_disabled, \
            event_drop_options, event_drop_value, event_drop_disabled, \
            dropdown_icon, sqa_drop_options, sqa_drop_value

    # === Update SQA plots ====================================================
    @app.callback(
        [Output('sqa-plot', 'figure'),
         Output('offcanvas', 'is_open', allow_duplicate = True),
         Output('postprocess-data', 'disabled')],
        [Input('memory-db', 'data'),
         Input('qa-charts-dropdown', 'value'),
         Input('data-dropdown', 'value'),
         Input('event-dropdown', 'value'),
         Input('re-render-sqa-flag', 'data')],
        prevent_initial_call = True
    )
    def update_sqa_plot(memory, sqa_view, selected_subject,
                        selected_event, re_render_sqa_flag):
        """Update the SQA plot based on the selected view and enable the
        'Postprocess' button."""

        # Get SQA data
        file = selected_subject
        event = selected_event
        if event:
            sqa = pd.read_csv(str(temp_path / f'{file}_{event}_SQA.csv'))
        else:
            sqa = pd.read_csv(str(temp_path / f'{file}_SQA.csv'))
        fs = int(memory['downsampled fs'])
        data_type = memory['data type']

        # Render cardio QA charts
        if data_type in ('ECG', 'PPG', 'BVP'):
            cardio_sqa = SQA.Cardio(fs)
            if sqa_view == 'missing':
                sqa_plot = cardio_sqa.plot_missing(sqa, title = file)
            elif sqa_view == 'artifact':
                sqa_plot = cardio_sqa.plot_artifact(sqa, title = file)
            else:
                sqa_plot = cardio_sqa.plot_missing(sqa, title = file)

        # Render EDA QA charts
        else:
            edaqa = SQA.EDA(fs)
            if sqa_view == 'validity':
                sqa_plot = edaqa.plot_validity(sqa, title = file)
            elif sqa_view == 'quality':
                sqa_plot = edaqa.plot_quality_metrics(sqa, title = file)
            else:
                sqa_plot = edaqa.plot_validity(sqa, title = file)

        return sqa_plot, False, False

    # === Update SQA table ====================================================
    @app.callback(
        [Output('summary-table', 'children'),
         Output('segment-dropdown', 'options'),
         Output('export-summary', 'disabled'),
         Output('export-mode', 'options'),
         Output('postprocess-export-mode', 'options')],
        [Input('memory-db', 'data'),
         Input('data-dropdown', 'value'),
         Input('event-dropdown', 'value'),
         Input('re-render-sqa-flag', 'data')],
        [State('data-dropdown', 'options'),
         State('event-dropdown', 'options'),
         State('event-segmentation-options', 'value'),
         State('toggle-filter', 'on'),
         State('seg-size', 'value')],
        prevent_initial_call = True
    )
    def update_sqa_table(memory, selected_subject, selected_event,
                         re_render_sqa_flag, all_subjects, all_events,
                         segment_event_by, filter_on, seg_size):
        """Update the SQA summary table and export batch options."""
        file = selected_subject
        event = selected_event
        data_type = memory['data type']
        file_type = memory['file type']
        signal_dur = memory['duration']
        segment_by_event = memory['segment by event']
        if any(x is None for x in (data_type, file_type, file)):
            raise PreventUpdate

        # Get SQA data
        fstem = f'{file}_{event}' if event else file
        sqa = pd.read_csv(str(temp_path / f'{fstem}_SQA.csv'))

        segments = sqa['Segment'].tolist()
        is_windowed = segment_event_by == 'windowed' and segment_by_event is True

        # Set the signal duration to the event duration if segmenting by
        # 'entire event'
        if event and not is_windowed:
            signal_dur = memory.get('event durations', {}).get(event, signal_dur)

        # Output signal quality table for cardiac data
        if data_type in ('ECG', 'PPG', 'BVP'):
            table, quality_summary = \
                _core.visualization._cardiac_summary_table(
                    sqa, duration = signal_dur, windowed = is_windowed,
                    window_size = seg_size)

        # Output signal quality table or EDA data
        else:
            eda = pd.read_csv(temp_path / f'{fstem}_EDA.csv')
            signal_col = 'Filtered' if filter_on else 'EDA'
            eda_signal = eda[signal_col].to_numpy()
            tonic_scl = compute_tonic_scl(eda_signal)

            # Generate EDA quality summary table
            table, quality_summary = _core.visualization._eda_summary_table(
                sqa, tonic_scl)

        # Create quality_summary.txt file(s)
        if selected_event:
            fnames = sorted(f'{s}_{e}' for s in all_subjects.values()
                            for e in all_events.values())
        else:
            fnames = sorted(all_subjects.values())
        for file in fnames:
            if not (temp_path / f'{file}_SQA.csv').exists():
                continue
            with open(str(temp_path / f'{file}_quality_summary.txt'), 'w') as f:

                # Add filename to the first line
                f.write(f'File: {file}\n')

                for label, value in quality_summary:
                    f.write(f'{label}: {value}\n')

        # Enable 'Batch' mode export
        segment_by_event = memory['segment by event']
        if file_type == 'batch' or segment_by_event:
            export_options = [
                {'label': 'Single', 'value': 'Single'},
                {'label': 'Batch', 'value': 'Batch', 'disabled': False}
            ]
        else:
            export_options = [
                {'label': 'Single', 'value': 'Single'},
                {'label': 'Batch', 'value': 'Batch', 'disabled': True}
            ]
        return table, segments, False, export_options, export_options

    # === Update signal plots =================================================
    @app.callback(
        [Output('raw-data', 'figure'),
         Output('segment-dropdown', 'value'),
         Output('prev-n-tooltip', 'is_open'),
         Output('next-n-tooltip', 'is_open'),
         Output('open-beat-editor', 'disabled', allow_duplicate = True),
         Output('beat-correction-status', 'data'),
         Output('beat-correction', 'hidden'),
         Output('accept-corrections', 'hidden'),
         Output('reject-corrections', 'hidden'),
         Output('revert-corrections', 'hidden'),
         Output('plot-displayed', 'data')],
        [Input('memory-db', 'data'),
         Input('segment-dropdown', 'value'),
         Input('data-dropdown', 'value'),
         Input('event-dropdown', 'value'),
         Input('prev-segment', 'n_clicks'),
         Input('next-segment', 'n_clicks'),
         Input('beat-correction', 'n_clicks'),
         Input('accept-corrections', 'n_clicks'),
         Input('reject-corrections', 'n_clicks'),
         Input('revert-corrections', 'n_clicks'),
         Input('be-edited-trigger', 'children')],
        [State('data-dropdown', 'options'),
         State('event-dropdown', 'options'),
         State('beat-correction-status', 'data'),
         State('seg-size', 'value'),
         State('toggle-filter', 'on'),
         State('segment-dropdown', 'options'),
         State('artifact-method', 'value'),
         State('artifact-tol', 'value'),
         State('temperature-load', 'data'),
         State('eda-valid-min', 'value'),],
        prevent_initial_call = True
    )
    def update_signal_plots(memory, selected_segment, selected_subject,
                            selected_event, prev_n, next_n, beat_correction_n,
                            accept_corrections_n, reject_corrections_n,
                            revert_corrections_n, beats_edited, all_subjects,
                            all_events, beat_correction_status, segment_size,
                            filter_on, segments, artifact_method, artifact_tol,
                            temp_data, eda_min):
        """Update the raw data plot based on the selected segment view."""
        if memory is None:
            raise PreventUpdate
        else:
            data_type = memory['data type']
            file_type = memory['file type']
            fs = int(memory['downsampled fs'])
            file = f'{selected_subject}_{selected_event}' if selected_event \
                else selected_subject

            # Get render data for primary signal
            render_subdir = render_dir / file
            signal = pd.read_csv(str(render_subdir / 'signal.csv'))
            y_axis_label = 'Filtered' if filter_on else data_type
            x_axis_label = 'Timestamp' if 'Timestamp' in signal.columns else \
                'Sample'
            ts_col = 'Timestamp' if 'Timestamp' in signal.columns else None

            # For visualization: Set segment size to length of entire signal
            # if segmenting by event and no segment size is given
            segment_by_event = memory['segment by event']
            if segment_by_event and segment_size is None:
                segment_size = len(signal)

            # Get ACC data if available
            try:
                acc = pd.read_csv(str(render_subdir / 'acc.csv'))
            except FileNotFoundError:
                acc = None

            trig = ctx.triggered_id

            # Reset selected_segment to 1 when new data is loaded or event changes
            if trig in ('memory-db', 'event-dropdown'):
                selected_segment = 1
                beat_correction_status = {}

            prev_tt_open = False
            next_tt_open = False

            # Handle prev/next segment clicks
            if trig == 'prev-segment':
                if selected_segment > 1:
                    selected_segment -= 1
                else:
                    prev_tt_open = True
            elif trig == 'next-segment':
                if selected_segment != max(segments):
                    selected_segment += 1
                else:
                    next_tt_open = True

            # Cardiac data workflow
            if data_type in ('ECG', 'PPG', 'BVP'):

                if beat_correction_status == {}:
                    if selected_event:
                        for subject in all_subjects:
                            for event in all_events:
                                beat_correction_status[
                                    f'{subject}_{event}'] = None
                    else:
                        for subject in all_subjects:
                            beat_correction_status[subject] = None

                def _save_temp_and_render(signal, file, data_type, fs, beats_ix,
                                          artifacts_ix, corrected_beats_ix = None):
                    beats_ix = np.array(beats_ix)
                    artifacts_ix = np.array(artifacts_ix)
                    if corrected_beats_ix is not None:
                        corrected_beats_ix = np.array(corrected_beats_ix)
                    signal.to_csv(str(temp_path / f'{file}_{data_type}.csv'), index = False)
                    ds_signal, ds_ibi, ds_ibi_corrected, _, _ = _core.io._downsample_data(
                        signal, fs, data_type, beats_ix, artifacts_ix, corrected_beats_ix)
                    ds_signal.to_csv(str(render_subdir / 'signal.csv'), index = False)
                    return ds_signal, ds_ibi, ds_ibi_corrected

                # Handle auto-corrections
                fs_full = memory['fs']
                ibi_corrected = None

                if trig == 'beat-correction':
                    signal = pd.read_csv(str(temp_path / f'{file}_{data_type}.csv'))
                    beats_ix = signal.loc[signal['Beat'] == 1].index.values
                    artifacts_ix = signal.loc[signal['Artifact'] == 1].index.values
                    signal, beats_ix_corrected, ibi_corrected = \
                        _core.beat_editing._correct_beats(signal, fs_full, beats_ix)
                    ibi_corrected.to_csv(str(temp_path / f'{file}_IBI_corrected.csv'), index = False)
                    signal, _, ibi_corrected = _save_temp_and_render(
                        signal, file, data_type, fs_full, beats_ix, artifacts_ix,
                        beats_ix_corrected)
                    ibi_corrected.to_csv(str(render_subdir / 'ibi_corrected.csv'), index = False)
                    beat_correction_status[file] = 'suggested'

                # Accept corrections and update signal and ibi files
                elif trig == 'accept-corrections':
                    beat_correction_status[file] = 'accepted'

                    # Update signal and ibi files to reflect accepted corrections
                    ibi = pd.read_csv(str(temp_path / f'{file}_IBI_corrected.csv'))
                    ibi.to_csv(str(temp_path / f'{file}_IBI.csv'), index = False)
                    ibi = pd.read_csv(str(render_subdir / 'ibi_corrected.csv'))
                    ibi.to_csv(str(render_subdir / 'ibi.csv'), index = False)
                    # ibi_corrected = None
                    signal = pd.read_csv(str(temp_path / f'{file}_{data_type}.csv'))
                    signal, beats_ix, artifacts_ix = \
                        _core.beat_editing._accept_beat_corrections(
                            signal, fs_full, artifact_method, artifact_tol)
                    signal, _, _ = _save_temp_and_render(
                        signal, file, data_type, fs_full, beats_ix, artifacts_ix)

                # Reject corrections and reset beat correction status
                elif trig == 'reject-corrections':
                    beat_correction_status[file] = None
                    # ibi_corrected = None

                # Revert corrections and update signal and ibi files to original
                elif trig == 'revert-corrections':
                    beat_correction_status[file] = None
                    # ibi_corrected = None
                    signal = pd.read_csv(str(temp_path / f'{file}_{data_type}.csv'))
                    signal, beats_ix, artifacts_ix = \
                        _core.beat_editing._revert_beat_corrections(
                            signal, fs_full, artifact_method, artifact_tol)
                    ibi = physioview.compute_ibis(
                        signal, fs_full, beats_ix, 'Timestamp')
                    ibi.to_csv(str(temp_path / f'{file}_IBI.csv'), index = False)
                    signal, ibi, _ = _save_temp_and_render(
                        signal, file, data_type, fs_full, beats_ix, artifacts_ix)
                    ibi.to_csv(str(render_subdir / 'ibi.csv'), index = False)

                # If beat correction status is suggested, render the corrected IBIs
                if beat_correction_status[file] == 'suggested':
                    ibi_corrected = pd.read_csv(str(render_subdir / 'ibi_corrected.csv'))

                # Get IBI data for rendering
                ibi = pd.read_csv(str(render_subdir / 'ibi.csv'))

                # Create the signal subplots with beat edits applied
                if beats_edited == file:
                    saved_dir = beat_editor_dir / 'saved'
                    data_dir = beat_editor_dir / 'data'

                    # Get for the subject's '_edit.json' file
                    edit_file = [p for p in (
                        data_dir / f'{file}_edit.json',
                        data_dir / 'batch' / f'{file}_edit.json'
                    ) if p.is_file()][0]
                    edits = pd.read_json(
                        str(saved_dir / f'{file}_edited.json'))
                    editor_data = pd.read_json(edit_file)

                    # Process beat edits
                    data_edited = physioview.process_beat_edits(
                        editor_data, edits)
                    data_edited_beats_ix = data_edited[
                        data_edited['Edited Beat'] == 1].index.values

                    # Recompute artifacts with edited beats
                    sqa = SQA.Cardio(fs)
                    artifacts_edited = sqa.identify_artifacts(
                        data_edited_beats_ix, method = artifact_method,
                        tol = artifact_tol)
                    if 'Artifact' in data_edited.columns:
                        artifact_col_pos = data_edited.columns.get_loc('Artifact')
                        del data_edited['Artifact']
                    else:
                        artifact_col_pos = len(data_edited.columns)
                    data_edited.insert(artifact_col_pos, 'Artifact', None)
                    data_edited.loc[artifacts_edited, 'Artifact'] = 1

                    # Recompute IBIs with edited beats for rendering
                    ibi_edited = physioview.compute_ibis(
                        data_edited, fs, data_edited_beats_ix, ts_col)

                    # Remove invalid IBIs + artifacts from any 'Unusable' portions
                    if 'Unusable' in data_edited.columns:
                        unusable_ix = data_edited[data_edited.Unusable == 1].index.values
                        breaks = np.where(np.diff(unusable_ix) > 1)[0]
                        if len(breaks) == 0:
                            starts = [unusable_ix[0]]
                            ends = [unusable_ix[-1]]
                        else:
                            starts = np.insert(unusable_ix[breaks + 1], 0, unusable_ix[0])
                            ends = np.append(unusable_ix[breaks], unusable_ix[-1])

                        unusable_bounds = list(zip(starts, ends))
                        for s, e in unusable_bounds:

                            # Get the last valid values before 'Unusable'
                            ibi_pre_ix = ibi_edited['IBI'].loc[:s-1].last_valid_index()
                            artif_pre_ix = data_edited['Artifact'].loc[:s-1].last_valid_index()
                            if ibi_pre_ix is not None:
                                ibi_edited.loc[ibi_pre_ix] = np.nan
                            if artif_pre_ix is not None:
                                data_edited.loc[artif_pre_ix, 'Artifact'] = np.nan

                            # Get the first valid values after 'Unusable'
                            ibi_post_ix = ibi_edited['IBI'].loc[e+1:].first_valid_index()
                            artif_post_ix = data_edited['Artifact'].loc[e+1:].first_valid_index()
                            if ibi_post_ix is not None:
                                ibi_edited.loc[ibi_post_ix] = np.nan
                            if artif_post_ix is not None:
                                data_edited.loc[artif_post_ix, 'Artifact'] = np.nan

                    # Save edited data
                    data_edited.to_csv(
                        str(temp_path / f'{file}_edited.csv'), index = False)

                    # Render updated signal plots
                    signal_plots = physioview.plot_signal(
                        signal = data_edited, signal_type = data_type,
                        axes = (x_axis_label, 'Signal'), fs = fs,
                        peaks_map = {data_type: 'Edited Beat'},
                        peaks_label = 'Edited Beat',
                        peaks_color = '#71b4eb',
                        edits_map = {data_type: {'Add': 'Added Beat',
                                                 'Unusable': 'Unusable'}},
                        artifacts_map = {data_type: 'Artifact'},
                        acc = acc, ibi = ibi_edited,
                        seg_number = selected_segment,
                        seg_size = segment_size)

                else:
                    overlay_corrected = beat_correction_status[file] == 'suggested'
                    correction_map = {data_type: 'Corrected'} if overlay_corrected else None

                    # Create cardiac signal subplots
                    signal_plots = physioview.plot_signal(
                        signal = signal, signal_type = data_type,
                        axes = (x_axis_label, y_axis_label), fs = fs,
                        peaks_map = {data_type: 'Beat'},
                        artifacts_map = {data_type: 'Artifact'},
                        correction_map = correction_map,
                        acc = acc, ibi = ibi,
                        ibi_corrected = ibi_corrected,
                        seg_number = selected_segment,
                        seg_size = segment_size)

                for trace in signal_plots.data:
                    if trace.name == y_axis_label and y_axis_label == 'Filtered':
                        trace.name = f'Filtered {data_type}'

                beat_correction_hidden = beat_correction_status[file] ==  'suggested' \
                    or beat_correction_status[file] == 'accepted'
                accept_corrections_hidden = beat_correction_status[file] != 'suggested'
                reject_corrections_hidden = beat_correction_status[file] != 'suggested'
                revert_corrections_hidden = beat_correction_status[file] != 'accepted'

            # Otherwise create the EDA signal subplots
            else:
                eda_subplots = {'EDA': ['Decomposed', y_axis_label, 'Tonic']}
                signal_types = [data_type]

                # Add temperature to subplots if data was provided
                # Fall back to the uploaded temperature store when the
                # uploaded data does not already contain it
                if temp_data is not None and 'Temp' not in signal.columns:
                    signal['Temp'] = temp_data['Temp']
                if 'Temp' in signal.columns:
                    signal_types.append('TEMP')
                    eda_subplots['TEMP'] = 'Temp'

                # Check whether SCRs were detected
                has_scr = 'SCR' in signal.columns

                # Create EDA subplots
                signal_plots = physioview.plot_signal(
                    signal = signal, signal_type = signal_types,
                    axes = (x_axis_label, eda_subplots),
                    fs = fs,
                    peaks_map = {data_type: 'SCR'} if has_scr else None,
                    # hline = eda_min, hline_name = 'Min. Valid EDA',
                    acc = acc, seg_number = selected_segment,
                    seg_size = segment_size)

                # Reorder traces in plot so SCRs are plotted on the phasic
                # component
                first_trace = None
                other_traces = []
                for trace in signal_plots.data:
                    if trace.name == 'Decomposed':
                        trace.name = 'Phasic'
                        other_traces.append(trace)
                    elif trace.name == y_axis_label:
                        if y_axis_label == 'Filtered':
                            trace.name = 'Filtered EDA'
                        trace.line.color = 'lightgrey'
                        first_trace = trace
                    elif trace.name == 'Tonic':
                        trace.line.dash = 'dash'
                        other_traces.append(trace)
                    else:
                        other_traces.append(trace)
                signal_plots.data = [first_trace] + other_traces if \
                    first_trace else other_traces

                beat_correction_hidden = False
                accept_corrections_hidden = True
                reject_corrections_hidden = True
                revert_corrections_hidden = True

            plot_displayed = True

            return [signal_plots, selected_segment, prev_tt_open, next_tt_open,
                    True, beat_correction_status, beat_correction_hidden,
                    accept_corrections_hidden, reject_corrections_hidden,
                    revert_corrections_hidden, plot_displayed]

    # === Open export summary modal ===========================================
    @app.callback(
        Output('export-modal', 'is_open'),
        [Input('export-summary', 'n_clicks'),
         Input('close-export', 'n_clicks'),
         Input('close-export2', 'n_clicks')],
        State('export-modal', 'is_open')
    )
    def toggle_export_modal(n1, cancel, done, is_open):
        """Open and close the Export Summary modal."""
        if n1 or cancel or done:
            return not is_open
        else:
            return is_open

    # === Download summary data ===============================================
    @callback(
        output = [
            Output('export-description', 'hidden'),
            Output('download-summary', 'data'),
            Output('export-confirm', 'hidden'),
            Output('export-modal-btns', 'hidden'),
            Output('export-close-btn', 'hidden')
        ],
        inputs = [
            Input('ok-export', 'n_clicks'),
            Input('close-export2', 'n_clicks'),
            State('export-mode', 'value'),
            State('export-type', 'value'),
            State('data-dropdown', 'value'),
            State('data-dropdown', 'options'),
            State('event-dropdown', 'value'),
            State('event-dropdown', 'options'),
            State('memory-db', 'data'),
        ],
        background = True,
        running = [
            (Output('export-progress-bar', 'style'),
             {'visibility': 'visible'}, {'visibility': 'hidden'}),
            (Output('ok-export', 'disabled'), True, False),
        ],
        cancel = [
            Input('close-export', 'n_clicks')
        ],
        progress = [
            Output('export-progress-bar', 'value'),
            Output('export-progress-bar', 'label')
        ],
        prevent_initial_call = True
    )
    def export_summary(set_progress, n, done, export_mode, export_type,
                       selected_subject, all_subjects, selected_event,
                       all_events, memory):
        """Export the SQA summary file and confirm the export."""
        if ctx.triggered_id in ('close-export', 'close-export2'):
            set_progress((0, ''))
            return [False, None, True, False, True]
        else:
            data_type = memory['data type']
            file_type = memory['file type']
            if export_mode == 'Single':
                file = f'{selected_subject}_{selected_event}' if selected_event \
                    else selected_subject
                files2export = [temp_path / f'{file}_SQA.csv']
                if data_type == 'BVP':  # if data is from the Empatica E4
                    files2export.extend([
                        temp_path / f'{file}_BVP.csv',
                        temp_path / f'{file}_IBI.csv',
                        temp_path / f'{file}_EDA.csv',
                        temp_path / f'{file}_quality_summary.txt'
                    ])
                elif data_type == 'Actiwave':
                    files2export.extend([
                        temp_path / f'{file}_ECG.csv',
                        temp_path / f'{file}_IBI.csv',
                        temp_path / f'{file}_quality_summary.txt'])
                elif data_type == 'PPG':
                    files2export.extend([
                        temp_path / f'{file}_PPG.csv',
                        temp_path / f'{file}_IBI.csv',
                        temp_path / f'{file}_quality_summary.txt'])
                elif data_type == 'ECG':
                    files2export.extend([
                        temp_path / f'{file}_ECG.csv',
                        temp_path / f'{file}_IBI.csv',
                        temp_path / f'{file}_quality_summary.txt'])
                elif data_type == 'EDA':
                    files2export.extend([
                        temp_path / f'{file}_EDA.csv',
                        temp_path / f'{file}_quality_summary.txt'])
                    if file_type == 'E4':
                        files2export.extend([
                            temp_path / f'{file}_TEMP.csv'
                        ])
                if (temp_path / f'{file}_ACC.csv').exists():
                    files2export.append(temp_path / f'{file}_ACC.csv')

            else:  # if export_mode == 'Batch'
                fnames = sorted([s + '_' + e for s in all_subjects.values()
                                 for e in all_events.values()])
                files2export = [temp_path / f'{f}_SQA.csv' for f in fnames]
                for f in fnames:
                    if data_type == 'BVP':  # if data is from the Empatica E4
                        files2export.extend([
                            temp_path / f'{f}_BVP.csv',
                            temp_path / f'{f}_IBI.csv',
                            temp_path / f'{f}_EDA.csv',
                            temp_path / f'{f}_quality_summary.txt'
                        ])
                    elif data_type == 'Actiwave':
                        files2export.extend([
                            temp_path / f'{f}_ECG.csv',
                            temp_path / f'{f}_IBI.csv',
                            temp_path / f'{f}_quality_summary.txt'
                        ])
                    elif data_type == 'PPG':
                        files2export.extend([
                            temp_path / f'{f}_PPG.csv',
                            temp_path / f'{f}_IBI.csv',
                            temp_path / f'{f}_quality_summary.txt'
                        ])
                    elif data_type == 'ECG':
                        files2export.extend([
                            temp_path / f'{f}_ECG.csv',
                            temp_path / f'{f}_IBI.csv',
                            temp_path / f'{f}_quality_summary.txt'
                        ])
                    elif data_type == 'EDA':
                        files2export.extend([
                            temp_path / f'{f}_EDA.csv',
                            temp_path / f'{f}_quality_summary.txt'])
                        if file_type == 'E4':
                            files2export.extend([
                                temp_path / f'{f}_TEMP.csv'
                            ])
                    if (temp_path / f'{f}_ACC.csv').exists():
                        files2export.append(temp_path / f'{f}_ACC.csv')

            files2export = [p for p in files2export if p.exists()]

            # Record timestamp of export
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            export = None

            # --- Zip export format ------------------------------------------
            if export_type == 'Zip':

                # Initialize a zip file for the preprocessed files
                output_zip = BytesIO()
                with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for i, file_path in enumerate(files2export):
                        file_name = file_path.name
                        with open(file_path, 'rb') as f:
                            zf.writestr(file_name, f.read())

                            # Update progress bar with each file
                            perc = (i + 1) / len(files2export) * 100
                            set_progress((perc, f'{perc:.0f}%'))
                            sleep(0.5)

                # Save zip file
                output_zip.seek(0)
                export = send_bytes(output_zip.getvalue(),
                                    f'sqa_summary_{current_time}.zip')

            # --- Excel export format ----------------------------------------
            elif export_type == 'Excel':

                # --- Batch mode ---------------------------------------------
                if export_mode == 'Batch':

                    # Initialize batch zip file to hold subjects' Excel files
                    batch_zip = BytesIO()
                    with zipfile.ZipFile(batch_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        subject_groups = defaultdict(list)
                        for f in files2export:
                            filename = Path(f).name

                            # Group files by subject
                            if filename.endswith('_quality_summary.txt'):
                                subject = filename.replace(
                                    '_quality_summary.txt', '')
                            else:
                                subject = filename.rsplit('_', 1)[0]

                            # subject = Path(f).name
                            subject_groups[subject].append(f)

                        # Set parameters for progress bar
                        total_files = len(files2export) + 1
                        n_processed = 1
                        perc = n_processed / total_files * 100
                        set_progress((perc, f'{perc:.0f}%'))

                        # Create Excel file for each subject's set of files
                        for i, (subj, subj_files) in enumerate(subject_groups.items()):
                            output_xls = _core.io._make_excel(subj_files)

                            # Update progress bar per file
                            n_processed += 1
                            perc = n_processed / total_files * 100
                            set_progress((perc, f'{perc:.0f}%'))

                            # Add subject's Excel file to batch zip
                            zf.writestr(f'{subj}_sqa_summary.xlsx', output_xls.read())

                    # Save zip file
                    batch_zip.seek(0)
                    export = send_bytes(
                        batch_zip.getvalue(), f'sqa_summary_{current_time}.zip')

                # --- Single file mode ---------------------------------------
                else:
                    # Create Excel file for a single subject
                    xls_workbook = _core.io._make_excel(
                        files2export, set_progress = set_progress)

                    # Save Excel file
                    export = send_bytes(
                        xls_workbook.getvalue(),
                        f'sqa_summary_{current_time}.xlsx'
                    )

            set_progress((100, '100%'))
            sleep(0.3)
            return [True, export, False, True, False]

    # === Enable OK summary export button =====================================
    @app.callback(
        Output('ok-export', 'disabled'),
        Input('export-type', 'value')
    )
    def enable_summary_export_button(export_type):
        if export_type is not None:
            return False
        return True

    # === Enable/Disable Beat Correction buttons ==============================
    @app.callback(
        [Output('beat-correction', 'disabled'),
         Output('revert-corrections', 'disabled')],
        [Input('plot-displayed', 'data'),
         Input('be-edited-trigger', 'children')],
        [State('data-dropdown', 'value'),
         State('data-types', 'value'),
         State('e4-data-types', 'value')]
    )
    def update_beat_correction_buttons(plot_displayed, beats_edited,
                                       selected_subject, dtype, e4_dtype):
        # If the subject has been edited, disable the beat correction button
        if beats_edited == selected_subject:
            return True, True
        elif plot_displayed is False or dtype == 'EDA' or e4_dtype == 'EDA':
            return True, True
        else:
            return False, False

    # ======================== BEAT EDITOR ELEMENTS ===========================
    # === Update Beat Editor button label and trigger on edit =================
    @app.callback(
        [Output('beat-editor-btn-label', 'children'),
         Output('open-beat-editor', 'style'),
         Output('be-edited-trigger', 'children')],
        [Input('ok-beat-edits', 'n_clicks'),
         Input('cancel-beat-edits', 'n_clicks'),
         Input('data-dropdown', 'value'),
         Input('event-dropdown', 'value'),
         State('beat-editor-modal', 'is_open'),
         State('data-dropdown', 'value'),
         State('event-dropdown', 'value'),
         State('be-edited-trigger', 'children')],
        prevent_initial_call = True
    )
    def reflect_beat_edits(n_apply, n_cancel, subject_dropdown, event_dropdown,
                           beat_editor_open, selected_subject,
                           selected_event, prev_beats_edited):
        """Update the Beat Editor button label, style, and trigger state
        when edits are detected in the saved file."""
        trig = ctx.triggered_id

        # Default button styling
        btn_label = 'Beat Editor'
        btn_style = {}
        beats_edited = None
        saved_dir = Path('beat-editor/saved')
        edits_stem = f'{selected_subject}_{selected_event}' if selected_event \
            else selected_subject
        edited_file = saved_dir / f'{edits_stem}_edited.json'

        if trig in ('data-dropdown', 'event-dropdown'):
            if edited_file.exists():
                btn_label = 'Beats Edited'
                btn_style = {'background': '#f1ab2a'}
                beats_edited = edits_stem
        elif trig == 'ok-beat-edits':
            if edited_file.exists() and _core.beat_editing._check_beat_editor_status():
                btn_label = 'Beats Edited'
                btn_style = {'background': '#f1ab2a'}
                beats_edited = edits_stem
        elif trig == 'cancel-beat-edits':
            # Keep beats_edited only if it was already set previously
            if prev_beats_edited == edits_stem:
                btn_label = 'Beats Edited'
                btn_style = {'background': '#f1ab2a'}
                beats_edited = edits_stem

        return btn_label, btn_style, beats_edited

    # === Open/Close Beat Editor modal ========================================
    @app.callback(
        [Output('beat-editor-modal', 'is_open'),
         Output('beat-editor-content', 'children'),
         Output('ok-beat-edits', 'disabled')],
        [Input('open-beat-editor', 'n_clicks'),
         Input('ok-beat-edits', 'n_clicks'),
         Input('cancel-beat-edits', 'n_clicks'),
         State('data-dropdown', 'value'),
         State('event-dropdown', 'value')],
        prevent_initial_call = True
    )
    def toggle_beat_editor(beat_editor_clicked, apply_beats_clicked,
                           beat_editor_cancel_clicked, selected_subject,
                           selected_event):
        """Open or close the Beat Editor modal."""
        clicked = ctx.triggered_id

        if clicked in ('cancel-beat-edits', 'ok-beat-edits'):
            return False, None, False

        data_dir = Path('beat-editor/data')
        batch_dir = data_dir / 'batch'

        try:
            # Determine the edit filename for the current selection
            if selected_subject and selected_event:
                edit_fstem = f'{selected_subject}_{selected_event}'
            elif selected_subject:
                edit_fstem = selected_subject
            else:
                edit_fstem = None

            if batch_dir.exists():
                # Move any current JSON file back to 'beat-editor/data/batch'
                current_data = list(data_dir.glob('*_edit.json'))
                for f in current_data:
                    if edit_fstem and f.name == f'{edit_fstem}_edit.json':
                        continue
                    dest = batch_dir / f.name
                    try:
                        dest.unlink()
                    except Exception:
                        pass
                    shutil.move(f, dest)

                # Move selected file's '_edit.json' to 'beat-editor/data'
                if edit_fstem:
                    src = batch_dir / f'{edit_fstem}_edit.json'
                    if src.exists():
                        dest = data_dir / src.name
                        try:
                            dest.unlink()
                        except Exception:
                            pass
                        shutil.move(src, dest)

            # Render Beat Editor modal content
            edit_jsons = list(data_dir.glob('*_edit.json'))
            if not edit_jsons:
                content = html.Span('No data available.')
                apply_disabled = True
            else:
                if _core.beat_editing._check_beat_editor_status():
                    content = html.Iframe(
                        id = 'beat-editor-iframe',
                        src = 'http://localhost:3000',
                        style = {'width': '100%', 'height': '525px',
                                 'border': 'none', 'overflow': 'hidden'},
                    )
                    apply_disabled = False if selected_subject else True
                else:
                    content = [
                        html.Span('Beat Editor is not running.'),
                        html.P([
                            'Check the ',
                            html.A(
                                'startup instructions',
                                href = (
                                    'https://physioview.readthedocs.io/en/latest/'
                                    'beat-editor-getting-started.html#'
                                    'launching-the-beat-editor'
                                ),
                                target = '_blank'
                            ), '.'
                        ])
                    ]
                    apply_disabled = True
        except:
            content = html.Span('No data available.')
            apply_disabled = True
        return True, content, apply_disabled

    # ======================= POSTPROCESESING MODAL ==========================
    # === Open/Close Postprocessing modal ====================================
    @app.callback(
        [Output('postprocess-modal', 'is_open', allow_duplicate = True),
         Output('postprocess-options', 'options')],
        [Input('postprocess-data', 'n_clicks'),
         Input('cancel-postprocess', 'n_clicks'),
         State('data-dropdown', 'value'),
         State('postprocess-modal', 'is_open'),
         State('memory-db', 'data')],
        prevent_initial_call = True
    )
    def toggle_postprocessing(postprocess_clicked, cancel_clicked,
                              selected_subject, is_open, memory):
        """Open and close the postprocessing modal."""

        # Base options with Interval Series disabled for EDA
        data_type = memory['data type']
        int_disabled = data_type == 'EDA'
        opts = [
            {'label': 'Raw and Cleaned Signal', 'value': 'signal_data'},
            {'label': 'Interval Series', 'value': 'interval_data',
             'disabled': int_disabled},
            {'label': 'Derived Features', 'value': 'features'},
            {'label': 'Signal Quality Metrics', 'value': 'sqa'}
        ]

        clicked = ctx.triggered_id
        if clicked == 'postprocess-data':
            if selected_subject is None:
                return is_open, opts
            return True, opts
        if clicked == 'cancel-postprocess':
            if is_open:
                return False, opts
            else:
                return is_open, opts
        return is_open, opts

    # === Select all output types ============================================
    @app.callback(
        [Output('postprocess-options', 'value'),
         Output('select-all', 'children')],
        [Input('select-all', 'n_clicks'),
         Input('postprocess-options', 'value'),
         State('postprocess-options', 'value'),
         State('memory-db', 'data')],
        prevent_initial_call = True
    )
    def select_all_output_types(n, selected, current_selection, memory):
        data_type = memory['data type']
        int_disabled = data_type == 'EDA'
        all_output_types = [
            {'label': 'Raw and Cleaned Signal', 'value': 'signal_data'},
            {'label': 'Interval Series', 'value': 'interval_data'},
            {'label': 'Derived Features', 'value': 'features'},
            {'label': 'Signal Quality Metrics', 'value': 'sqa'}
        ]

        # Exclude interval_data from 'Select All' values if disabled
        if int_disabled:
            all_values = [opt['value'] for opt in all_output_types
                          if opt['value'] != 'interval_data']
        else:
            all_values = [opt['value'] for opt in all_output_types]

        trig = ctx.triggered_id

        if trig == 'select-all':
            if set(current_selection) == set(all_values):
                return [], 'Select All'
            else:
                return all_values, 'Deselect All'
        elif trig == 'postprocess-options':
            if set(selected or []) == set(all_values):
                return selected, 'Deselect All'
            else:
                return selected, 'Select All'

    # === Enable postprocessing data parameterization ========================
    @app.callback(
        [Output('feature-window-size', 'disabled'),
         Output('feature-step-size', 'disabled'),
         Output('postprocess-params-container', 'style')],
        Input('postprocess-options', 'value'),
        prevent_initial_call = True
    )
    def enable_postprocessing_parameters(selected_output_types):
        if 'features' in selected_output_types:
            return False, False, {'opacity': 1.0, 'fontStyle': 'normal'}
        else:
            return True, True, {'opacity': 0.5, 'fontStyle': 'italic'}

    # === Run Postprocessing pipeline ========================================
    @callback(
        output = [
            Output('postprocess-modal', 'is_open', allow_duplicate = True),
            Output('postprocess-done-toast', 'is_open'),
            Output('postprocess-error-toast', 'is_open'),
            Output('download-postprocess', 'data'),
        ],
        inputs = [
            Input('ok-postprocess', 'n_clicks'),
        ],
        state = [
            State('memory-db', 'data'),
            State('data-dropdown', 'value'),
            State('data-dropdown', 'options'),
            State('event-dropdown', 'value'),
            State('event-dropdown', 'options'),
            State('postprocess-options', 'value'),
            State('feature-window-size', 'value'),
            State('feature-step-size', 'value'),
            State('postprocess-export-mode', 'value'),
            State('postprocess-export-type', 'value')
        ],
        background = True,
        running = [
            (Output('postprocess-progress-bar', 'style'),
             {'visibility': 'visible'}, {'visibility': 'hidden'}),
        ],
        cancel = [
            Input('cancel-postprocess', 'n_clicks')
        ],
        progress = [
            Output('postprocess-progress-bar', 'value'),
            Output('postprocess-progress-bar', 'label')
        ],
        prevent_initial_call = True
    )
    def postprocess_data(set_progress, n, memory, selected_subject,
                         all_subjects, selected_event, all_events, outputs,
                         window_size, step_size, export_mode, export_fmt):
        """Run the data postprocessing pipeline based on the user-selected
        output types and postprocessing export mode and format."""
        if len(outputs) == 0 or (export_mode is None) or (export_fmt is None):
            return True, False, True, None

        # Reset progress
        set_progress((0, '0%'))

        # Get data parameters
        data_type = memory['data type']
        filename = memory['filename']
        fs_full = int(memory['fs'])  # original fs
        subjects = list(all_subjects.values())
        events = sorted(all_events.values()) if all_events else []

        # Set flags for postprocessing output types
        want_signal = 'signal_data' in outputs
        want_int = 'interval_data' in outputs
        want_feats = 'features' in outputs
        want_sqa = 'sqa' in outputs

        # Initialize file counter for progress tracking
        signals = ['ACC', 'ECG', 'BVP', 'PPG', 'EDA']
        signal_files = [
            f for f in temp_path.iterdir()
            if f.is_file() and any(f.stem.endswith(sig) for sig in signals)]
        n_signals = len(signal_files)
        n_files = sum(
            {'signal_data': n_signals, 'interval_data': 1,
             'features': 1, 'sqa': 2}[o] for o in outputs
            if o in {'signal_data', 'interval_data', 'features', 'sqa'})

        # Initialize progress updater
        progress_done = 0
        if data_type in ('ECG', 'PPG', 'BVP'):
            total_progress = len(subjects) * n_files + 2
        else:
            total_progress = len(subjects) * n_files + 1
        def _update_progress(units = 1):
            nonlocal progress_done
            progress_done += units
            perc = (progress_done / max(total_progress, 1)) * 100
            set_progress((perc, f'{perc:.0f}%'))
            sleep(0.3)
        _update_progress()

        # Helper function for postprocessing one subject's file(s)
        def _postprocess_one(s: str) -> list[Path]:
            """Postprocess data for a selected subject and aggregate
            user-selected output files into a list of Path objects.
            :param s: The selected subject.
            """
            out = []

            raw_path = temp_path / f'{s}_{data_type}.csv'
            cleaned_path = temp_path / f'{s}_{data_type}_cleaned.csv'

            # Prefer cleaned data if it exists
            in_path = cleaned_path if cleaned_path.exists() else raw_path
            if not in_path.exists():
                raise FileNotFoundError(
                    f'Missing input ECG file for {s}: {raw_path} or '
                    f'{cleaned_path}')
            data = pd.read_csv(in_path)

            # Get timestamp/sample column
            ts_col = 'Timestamp' if 'Timestamp' in data.columns else \
                'Sample'

            # Rename auto corrected beats column
            if 'Original Beat' in data.columns:
                data.rename(columns = {'Beat': 'Auto Corrected Beat'},
                            inplace = True)

            has_edits = False
            has_corrections = False
            if data_type in ('ECG', 'BVP', 'PPG'):

                # ---------------- Process Edited Cardiac Data ---------------
                edited_file = temp_path / f'{s}_edited.csv'
                has_edits = edited_file.exists()
                corrected_file = temp_path / f'{s}_IBI_corrected.csv'
                has_corrections = corrected_file.exists()

                # Get sampling rate of Beat Editor data
                beat_editor_fs = int(memory['downsampled fs'])  # ~250

                if has_edits:

                    # Get indices of beat edits
                    edited = pd.read_csv(str(edited_file))
                    edited_beats_ix = edited[edited.get(
                        'Edited Beat').eq(1)].index.values

                    # Set default indices
                    deletions_ix = np.array([], dtype = int)
                    additions_ix = np.array([], dtype = int)
                    unusable_ix = np.array([], dtype = int)

                    # Map edited beats to original sampling rate
                    mapped_edits_ix = _core.beat_editing._map_beat_edits(
                        edited_beats_ix, beat_editor_fs, fs_full)
                    if 'Deleted Beat' in edited.columns:
                        deletions_ix = edited[edited.get(
                            'Deleted Beat').eq(1)].index.values
                    mapped_deletions_ix = _core.beat_editing._map_beat_edits(
                        deletions_ix, beat_editor_fs, fs_full)
                    if 'Added Beat' in edited.columns:
                        additions_ix = edited[edited.get(
                            'Added Beat').eq(1)].index.values
                    mapped_additions_ix = _core.beat_editing._map_beat_edits(
                        additions_ix, beat_editor_fs, fs_full)
                    if 'Unusable' in edited.columns:
                        unusable_ix = edited[edited.get(
                            'Unusable').eq(1)].index.values
                    mapped_unusable_ix = _core.beat_editing._map_beat_edits(
                        unusable_ix, beat_editor_fs, fs_full)

                    # Record edited beats in original data
                    data.loc[mapped_edits_ix, 'Edited Beat'] = 1
                    if mapped_deletions_ix.size:
                        data.loc[mapped_deletions_ix, 'Deleted Beat'] = 1
                    if mapped_additions_ix.size:
                        data.loc[mapped_additions_ix, 'Added Beat'] = 1
                    if mapped_unusable_ix.size:
                        data.loc[mapped_unusable_ix, 'Unusable'] = 1

                        # Add contiguous annotations for 'Unusable' portions
                        ratio = fs_full / beat_editor_fs
                        k = int(round(ratio))
                        if abs(ratio - k) < 1e-9 and k > 1:
                            starts = mapped_unusable_ix.astype(int)
                            blocks = starts[:, None] + np.arange(k, dtype = int)
                            blocks = blocks[blocks < len(data)]
                            data.loc[np.unique(blocks.ravel()), 'Unusable'] = 1
                        else:
                            starts = np.floor(
                                mapped_unusable_ix * ratio).astype(int)
                            ends = np.ceil(
                                (mapped_unusable_ix + 1) * ratio).astype(int)
                            ends = np.maximum(ends, starts + 1)
                            parts = np.array([], dtype = int)
                            for start, end in zip(starts, ends):
                                part = np.arange(start, min(end, len(data)), dtype = int)
                                parts = np.concatenate([parts, part])
                            if len(parts) > 0:
                                full = np.unique(parts)
                                data.loc[full, 'Unusable'] = 1

                    # Reposition columns for clarity
                    if 'Original Beat' in data.columns:
                        beats_col = 'Original Beat'
                    else:
                        data.rename(columns = {'Beat': 'Original Beat'},
                                    inplace = True)
                        beats_col = 'Original Beat'
                    col_order = ['Segment', ts_col, data_type, 'Filtered',
                                 beats_col, 'Artifact', 'Auto Corrected Beat',
                                 'Deleted Beat',  'Added Beat',
                                 'Unusable', 'Edited Beat']
                    order = [c for c in col_order if c in data.columns]
                    data = data[order]

                    # Rewrite to temp_path
                    data.to_csv(cleaned_path, index = False)

            # ------------------- Raw and Cleaned Data -------------------
            if want_signal:
                sig_files = [p for p in signal_files if s in str(p)]
                for sig_path in sig_files:
                    if sig_path.stem.endswith(('ECG', 'PPG', 'BVP')):
                        cleaned = temp_path / f'{s}_{data_type}_cleaned.csv'
                        if cleaned.exists():
                            out.append(cleaned)
                        else:
                            data = pd.read_csv(sig_path)
                            if 'Original Beat' in data.columns:
                                data.rename(columns = {
                                    'Beat': 'Auto Corrected Beat'},
                                    inplace = True)
                                beats_col = 'Original Beat'
                            else:
                                beats_col = 'Beat'
                            col_order = ['Segment', ts_col, data_type,
                                         'Filtered', beats_col, 'Artifact',
                                         'Auto Corrected Beat', 'Edited Beat']
                            order = [c for c in col_order if c in data.columns]
                            data = data[order]
                            data.to_csv(sig_path, index = False)
                            out.append(sig_path)
                    else:
                        out.append(sig_path)

            # ------------ Interval Series / Derived Features ------------
            if want_int or want_feats:

                # Cardiac feature extraction
                if data_type in ('ECG', 'PPG', 'BVP'):

                    # Recompute IBIs with edited beats
                    if has_edits:
                        edited_ibi = physioview.compute_ibis(
                            data, fs_full, mapped_edits_ix, ts_col)

                        # Remove first IBI after each 'unusable' portion
                        if 'Unusable' in data.columns:
                            unus_ix = data[data.Unusable == 1].index.values
                            diffs = np.diff(unus_ix)
                            unus_ends = unus_ix[np.where(diffs > 1)[0]]
                            unus_ends = np.append(unus_ends, unus_ix[-1])
                            edited_ibi_ix = edited_ibi[
                                ~pd.isna(edited_ibi.IBI)].index.values
                            rem = []
                            for ix in unus_ends:
                                pos = np.searchsorted(
                                    edited_ibi_ix, ix, side = 'right')
                                if pos < len(edited_ibi_ix):
                                    rem.append(edited_ibi_ix[pos])
                            edited_ibi.loc[rem, 'IBI'] = np.nan

                        # Rewrite IBI data to temp_path
                        edited_ibi.to_csv(str(temp_path / f'{s}_IBI.csv'),
                                   index = False)

                    # Rewrite corrected IBI data to temp_path if available
                    else:
                        if has_corrections:
                            corrected_ibi = pd.read_csv(
                                temp_path / f'{s}_IBI_corrected.csv')
                            corrected_ibi.to_csv(
                                str(temp_path / f'{s}_IBI.csv'), index = False)

                    if want_int:
                        p = temp_path / f'{s}_IBI.csv'
                        out.append(p)

                    if want_feats:
                        ibi = pd.read_csv(str(temp_path / f'{s}_IBI.csv'))
                        ibi[ts_col] = pd.to_datetime(
                            ibi[ts_col], errors = 'coerce')
                        ibi = ibi.dropna(subset = [ts_col]).set_index(ts_col)
                        ibi_series = ibi['IBI'].dropna().sort_index()

                        if ibi_series.empty or len(ibi_series.index) < 2:
                            hrv = pd.DataFrame()
                        else:
                            # Set window length based on available IBI data
                            dur_sec = (ibi_series.index.max() - ibi_series.index.min()).total_seconds()
                            dur_sec = max(dur_sec, 0)
                            window_len = min(int(window_size), int(np.floor(dur_sec)))
                            step_len = min(int(step_size), window_len)
                            hrv = get_hrv_features(
                                data = ibi_series,
                                window_length = window_len,
                                window_step_size = step_len,
                                domains = ['td', 'fd', 'nl', 'stat'],
                                threshold = 0.5, clean_data = False)

                        # Write HRV data to temp_path
                        p = temp_path / f'{s}_HRV.csv'
                        hrv.to_csv(str(p))
                        out.append(p)

                # EDA feature extraction
                elif data_type == 'EDA':
                    eda_features = compute_features(
                        data['EDA'], fs_full, window_size, step_size)

                    # Write EDA features to temp_path
                    p = temp_path / f'{s}_Features.csv'
                    eda_features.to_csv(str(p), index = False)
                    out.append(p)

            # ------------------ Signal Quality Metrics ------------------
            if want_sqa:
                if data_type in ('ECG', 'PPG', 'BVP'):
                    sqa_txt_path = temp_path / f'{s}_quality_summary.txt'
                    c = 'YES' if 'Auto Corrected' in data.columns else 'NO'
                    e = 'YES' if has_edits else 'NO'
                    with open(sqa_txt_path, 'a') as f:
                        f.write(f'\nAuto corrected: {c}')
                        f.write(f'\nEdited: {e}')
                for p in [temp_path / f'{s}_SQA.csv',
                          temp_path / f'{s}_quality_summary.txt']:
                    if p.exists():
                        out.append(p)
            return out

        # Get all files for export
        if export_mode.lower() == 'single':
            selected_data = f'{selected_subject}_{selected_event}' if selected_event \
                else selected_subject
            files2make = _postprocess_one(selected_data)
        elif export_mode.lower() == 'batch':
            files2make = []
            all_data = []
            for s in subjects:
                if events:
                    for event in events:
                        stem = f'{s}_{event}'
                        all_data.append(stem)
                        files2make.append(_postprocess_one(stem))
                else:
                    all_data.append(s)
                    files2make.append(_postprocess_one(s))
        _update_progress()  # add progress after file aggregation

        # Write data in requested format
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        buf = BytesIO()
        if export_fmt.lower() == 'excel':
            if export_mode.lower() == 'single':
                ext = 'xlsx'
                buf = _core.io._make_excel(
                    files2make, set_progress = set_progress,
                    progress_start = progress_done,
                    progress_total = total_progress)
            elif export_mode.lower() == 'batch':
                ext = 'zip'
                with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for i, stem in enumerate(all_data):
                        files = [f for f in files2make[i]]
                        xls_out = _core.io._make_excel(files)
                        zf.writestr(f'{stem}_processed.xlsx',
                                    xls_out.getvalue())
                        _update_progress()  # add progress per subject
        elif export_fmt.lower() == 'zip':  # if 'zip' format
            ext = 'zip'
            if export_mode.lower() == 'single':
                buf = _core.io._make_zip(
                    files2make, set_progress = set_progress,
                    progress_start = progress_done,
                    progress_total = total_progress)
            elif export_mode.lower() == 'batch':
                files = [f for sub in files2make for f in sub]
                buf = _core.io._make_zip(
                    files, set_progress = set_progress,
                    progress_start = progress_done,
                    progress_total = total_progress)

        # Write to disk
        if len(buf.getvalue()) > 0:
            buf.seek(0)
            export = send_bytes(
                lambda f: f.write(buf.getvalue()),
                f'{Path(filename).stem}_{current_time}.{ext}'
            )
        set_progress((100, '100%'))
        sleep(0.3)

        # Postprocessing finished; close the modal and show the toast
        return False, True, False, export

    # ========= Clientside callback to auto-focus Beat Editor iFrame =========
    app.clientside_callback(
        """
        function(is_open) {
            const KEY = "__beat_editor_focus_observer";
            const observer = new MutationObserver((_mutations, obs) => {
                const ifr = document.getElementById('beat-editor-iframe');
                ifr.focus();
                obs.disconnect();
            });
            
            observer.observe(document.body, { childList: true, subtree: true });
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('beat-editor-iframe-listener', 'data'),
        Input('beat-editor-modal', 'is_open'),
        prevent_initial_call = True
    )