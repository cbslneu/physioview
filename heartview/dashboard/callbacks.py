from dash import html, Input, Output, State, ctx, callback
from dash.exceptions import PreventUpdate
from dash.dcc import send_bytes
from heartview import heartview
from heartview.pipeline import ACC, SQA
from heartview.dashboard import utils
from flirt.hrv import get_hrv_features
from os import makedirs, stat
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

def get_callbacks(app):
    """Attach callback functions to the dashboard app."""

    root = Path(__file__).resolve().parents[2]
    temp_path = root / 'temp'
    render_dir = temp_path / '_render'
    beat_editor_dir = root / 'beat-editor'

    # ============================= DATA UPLOAD ===============================
    du.configure_upload(app, str(temp_path), use_upload_id = True)
    @du.callback(
        output = [
            Output('file-check', 'children'),
            Output('run-data', 'disabled'),
            Output('configure', 'disabled'),
            Output('memory-load', 'data'),
        ],
        id = 'dash-uploader'
    )
    def db_get_file_types(filenames):
        """Save the data type to the local memory depending on the file
        type."""
        if not filenames:
            # raise PreventUpdate
            return [], True, True, None

        session_path = Path(filenames[0]).parent
        latest_file = sorted(
            session_path.iterdir(), key = lambda f: -stat(f).st_mtime)[0]
        filename = str(latest_file)

        # Default visibility
        disable_run = True
        disable_configure = True

        ext = filenames[0].lower().rsplit('.', 1)[-1]
        if ext == 'edf':
            if utils._check_edf(filenames[0]) == 'ECG':
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
                    extract_dir = str(temp_path / session_path / 'batch')
                    makedirs(extract_dir, exist_ok = True)
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
            raise PreventUpdate

        # Clear Beat Editor directories
        utils._clear_edits()

        return [file_check, disable_run, disable_configure, data]

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
        configs = utils._load_config(cfg_file[0])
        return configs

    # ======================== ENABLE DATA RESAMPLING =========================
    @app.callback(
        [Output('resample', 'hidden'),
         Output('resampling-rate', 'disabled'),
         Output('cardio-preprocessing', 'hidden'),
         Output('beat-detectors', 'options'),
         Output('beat-detectors', 'value')],
        [Input('data-types', 'value'),
         Input('toggle-resample', 'on'),
         State('memory-load', 'data')],
        prevent_initial_call = True
    )
    def db_enable_dtype_specific_parameters(dtype, toggle_on, loaded_data):
        """Enable parameters specific to data types."""
        cardio_preprocess_hidden = True
        resample_hidden = True
        resample_disabled = True
        beat_detectors = []
        default_bd = None
        data_source = loaded_data['source']
        if dtype == 'EDA':
            resample_hidden = False
            if toggle_on is True:
                resample_disabled = False
            else:
                resample_disabled = True
        if dtype in ('PPG', 'ECG') or data_source in ('csv', 'E4', 'Actiwave'):
            cardio_preprocess_hidden = False
            if dtype == 'PPG' or data_source == 'E4':
                beat_detectors = [
                    {'label': 'Elgendi et al. (2013)', 'value': 'erma'},
                    {'label': 'Van Gent et al. (2018)', 'value': 'adaptive_threshold'}]
                default_bd = 'adaptive_threshold'
            else:
                beat_detectors = [
                    {'label': 'Manikandan & Soman (2012)', 'value': 'manikandan'},
                    {'label': 'Engels & Zeelenberg (1979)', 'value': 'engzee'},
                    {'label': 'Nabian et al. (2018)', 'value': 'nabian'},
                    {'label': 'Pan & Tompkins (1985)', 'value': 'pantompkins'}
                ]
                default_bd = 'manikandan'
        return [resample_hidden, resample_disabled,
                cardio_preprocess_hidden, beat_detectors, default_bd]

    # =================== POPULATE PARAMETERIZATION FIELDS ====================
    @app.callback(
        [Output('setup-data', 'hidden'),
         Output('preprocess-data', 'hidden'),
         Output('segment-data', 'hidden'),
         Output('data-type-container', 'hidden'),     # data type
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
         Output('sampling-rate', 'value'),
         Output('seg-size', 'value'),
         Output('artifact-method', 'value'),
         Output('artifact-tol', 'value'),
         Output('toggle-filter', 'on')],
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
        hide_setup = False
        hide_preprocess = False
        hide_segsize = False
        hide_data_types = False
        hide_data_vars = False
        hide_variable_error = True

        # Default parameter values
        base_headers = ['<Var>', '<Var>']
        drop_values = [base_headers[:] for _ in range(5)]
        filter_on = False
        artifact_method = 'cbd'
        artifact_tol = 1
        seg_size = 60
        fs = 500
        dtype = None

        if loaded == 'memory-load':

            # -- device sources ----------------------------------------------
            if memory['source'] == 'Actiwave':
                hide_setup = True
                hide_data_types = True
                hide_data_vars = True
                if toggle_config_on:
                    seg_size = configs['segment size']
                    fs = configs['sampling rate']
                    dtype = configs['data type']

            elif memory['source'] == 'E4':
                hide_setup = True
                hide_data_types = True
                hide_data_vars = True
                fs = 64
                if toggle_config_on:
                    seg_size = configs['segment size']
                    fs = configs['sampling rate']
                    dtype = configs['data type']

            # -- csv sources -------------------------------------------------
            elif memory['source'] == 'csv':
                if toggle_config_on:
                    pass
                else:
                    headers = utils._get_csv_headers(memory['filename'])
                    base_headers = headers
                    drop_values = [headers[:] for _ in range(5)]

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
                        zfiles = [f for f in zf.namelist() if '/' in f and
                                  not f.startswith('__MACOSX/') and
                                  not f.endswith('.DS_Store') and
                                  '/._' not in f and not f.endswith('/')]

                        for f in zfiles:
                            fname = Path(f).name
                            if utils._check_csv(f):
                                extracted_path = zf.extract(
                                    f, path = str(extract_dir))

                                # Move to root 'temp' directory
                                root_temp = Path(extract_dir) / fname
                                shutil.move(extracted_path, root_temp)

                                # Get CSV headers
                                hdrs = utils._get_csv_headers(str(root_temp))
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
                        drop_values = [headers[:] for _ in range(5)]

        elif loaded == 'config-memory':
            device = configs['source']
            dtype = configs['data type']
            seg_size = configs['segment size']
            fs = configs['sampling rate']
            artifact_method = configs['artifact identification method']
            artifact_tol = configs['artifact tolerance']
            filter_on = configs['filters']

            if device in ('E4', 'Actiwave'):
                hide_setup = hide_data_types = hide_data_vars = True
                base_headers = []
                drop_values = [[] for _ in range(5)]
            else:
                headers = list(configs['headers'].values())
                base_headers = headers
                drop_values = [[h for h in headers if h is not None] for _
                               in range(5)]

        dropdown_options = [{'label': h, 'value': h} for h in base_headers
                            if h is not None]

        return (
            hide_setup, hide_preprocess, hide_segsize, hide_data_types, dtype,
            hide_data_vars, hide_variable_error,

            # variable dropdowns
            dropdown_options, drop_values[0],
            dropdown_options, drop_values[1],
            dropdown_options, drop_values[2],
            dropdown_options, drop_values[3],
            dropdown_options, drop_values[4],

            fs, seg_size, artifact_method, artifact_tol, filter_on
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
         State('config-filename', 'value')],
        prevent_initial_call = True
    )
    def write_confirm_config(n, data, dtype, fs, d1, d2, d3, d4, d5,
                             seg_size, artifact_method, artifact_tol,
                             filter_on, filename):
        """Export the configuration file."""
        if n:
            headers = None
            device = data['source'] if data['source'] != 'csv' else 'Other'
            if device == 'Actiwave':
                actiwave = heartview.Actiwave(data['filename'])
                fs = actiwave.get_ecg_fs()
                dtype = 'ECG'
            elif device == 'E4':
                E4 = heartview.Empatica(data['filename'])
                fs = E4.get_bvp().fs
                dtype = 'BVP'
            else:
                headers = {
                    'Time/Sample': d1,
                    'Signal': d2,
                    'X': d3,
                    'Y': d4,
                    'Z': d5}
            json_object = utils._create_configs(
                device, dtype, fs, seg_size, artifact_method, artifact_tol,
                filter_on, headers)
            download = {'content': json_object, 'filename': f'{filename}.json'}
            return [download, 1]

    # ============================= RUN PIPELINE ==============================
    @callback(
        output = [
            Output('dtype-validator', 'is_open'),
            Output('mapping-validator', 'is_open'),
            Output('memory-db', 'data')
        ],
        inputs = [
            Input('run-data', 'n_clicks'),
            Input('close-dtype-validator', 'n_clicks'),
            Input('close-mapping-validator', 'n_clicks'),
            State('memory-load', 'data'),
            State('data-types', 'value'),
            State('sampling-rate', 'value'),
            State('data-type-dropdown-1', 'value'),
            State('data-type-dropdown-2', 'value'),
            State('data-type-dropdown-3', 'value'),
            State('data-type-dropdown-4', 'value'),
            State('data-type-dropdown-5', 'value'),
            State('beat-detectors', 'value'),
            State('seg-size', 'value'),
            State('artifact-method', 'value'),
            State('artifact-tol', 'value'),
            State('toggle-filter', 'on')
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
    def run_pipeline(set_progress, n, close_dtype_err, close_mapping_err,
                     load_data, dtype, fs, d1, d2, d3, d4, d5, beat_detector,
                     seg_size, artifact_method, artifact_tol, filt_on):
        """Read Actiwave Cardio, Empatica E4, or CSV-formatted data, save
        the data to the local memory, and load the progress spinner."""

        dtype_error = False
        map_error = False

        if ctx.triggered_id in ('close-dtype-validator',
                                'close-mapping-validator'):
            return False, False, None

        if ctx.triggered_id == 'run-data':

            # Create '_render' folder
            render_dir.mkdir(parents = True, exist_ok = True)

            file_type = load_data['source']
            if file_type == 'E4':
                dtype = 'BVP'
            elif file_type == 'Actiwave':
                dtype = 'ECG'
            else:
                if dtype is None:
                    dtype_error = True
                    return dtype_error, map_error, None
                elif d2 is None:
                    map_error = True
                    return dtype_error, map_error, None

            filepath = load_data['filename']
            filename = Path(filepath).name  # e.g., "example.csv"
            file = Path(filepath).stem
            preprocessed = {}

            # Handle batch files
            if file_type == 'batch':
                batch_file = Path(filepath)
                session_path = batch_file.parent
                batch_dir = session_path / 'batch'
                batch = sorted([
                    f for f in batch_dir.iterdir()
                    if f.is_file() and not f.name.startswith('.') and
                       f.suffix == '.csv'])
                ds = True  # enable downsampling

                # Set progress bar total
                total_progress = len(batch) + 1
                perc = (1 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))

                # Preprocess each file in the batch
                for idx, f in enumerate(batch):
                    fname = f.stem
                    if d1 is None:
                        # If no acceleration data are provided
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data = utils._setup_data_samples(
                                f, dtype, [d2])
                        else:
                            raw = utils._setup_data_samples(
                                f, dtype, [d2, d3, d4, d5])
                            data = raw[['Sample', dtype]].copy()
                            acc = raw[['Sample', 'X', 'Y', 'Z']].copy()
                    else:
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data = utils._setup_data_ts(
                                f, dtype, [d1, d2])
                        else:
                            raw = utils._setup_data_ts(
                                f, dtype, [d1, d2, d3, d4, d5])
                            data = raw[['Timestamp', dtype]].copy()
                            acc = raw[['Timestamp', 'X', 'Y', 'Z']].copy()

                    # Preprocess any acceleration data
                    try:
                        acc['Magnitude'] = ACC.compute_magnitude(
                            acc['X'], acc['Y'], acc['Z'])
                        acc.to_csv(
                            str(temp_path / f'{fname}_ACC.csv'),
                            index = False)
                    except:
                        acc = None

                    (preprocessed_data, ibi, metrics, ds_data, ds_ibi,
                     ds_acc, ds_fs) = utils._preprocess_cardiac(
                        data, dtype, fs, seg_size, beat_detector,
                        artifact_method, artifact_tol, filt_on,
                        acc_data = acc, downsample = ds)

                    # Write preprocessed data to 'temp' folder
                    preprocessed_data.to_csv(
                        str(temp_path / f'{fname}_{dtype}.csv'), index = False)
                    ibi.to_csv(
                        str(temp_path / f'{fname}_IBI.csv'), index = False)
                    metrics.to_csv(
                        str(temp_path / f'{fname}_SQA.csv'), index = False)

                    # Write downsampled data to '_render' folder
                    render_subdir = render_dir / fname
                    render_subdir.mkdir(parents = True, exist_ok = True)
                    ds_data.to_csv(str(render_subdir / 'signal.csv'),
                                   index =  False)
                    ds_ibi.to_csv(str(render_subdir / 'ibi.csv'),
                                  index = False)
                    if ds_acc is not None:
                        ds_acc.to_csv(str(render_subdir / 'acc.csv'),
                                      index = False)

                    perc = ((idx + 2) / total_progress) * 100
                    set_progress((perc * 100, f'{perc:.0f}%'))

            elif file_type in ('Actiwave', 'E4', 'csv'):

                # Handle Actiwave files
                if file_type == 'Actiwave':
                    ds = True  # enable downsampling
                    actiwave = heartview.Actiwave(filepath)
                    actiwave_data = actiwave.preprocess(time_aligned = True)
                    data = actiwave_data[['Timestamp', dtype]].copy()
                    acc = actiwave_data[['Timestamp', 'X', 'Y', 'Z']].copy()
                    acc.to_csv(
                        str(temp_path / f'{file}_ACC.csv'), index = False)
                    fs = actiwave.get_ecg_fs()

                # Handle other ECG/PPG sources
                elif file_type == 'csv':
                    ds = True  # enable downsampling
                    # If no timestamps are provided
                    if d1 is None:
                        # If no acceleration data are provided
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data = utils._setup_data_samples(
                                filepath, dtype, [d2])
                        else:
                            raw = utils._setup_data_samples(
                                filepath, dtype, [d2, d3, d4, d5])
                            data = raw[['Sample', dtype]].copy()
                            acc = raw[['Sample', 'X', 'Y', 'Z']].copy()
                    else:
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data = utils._setup_data_ts(
                                filepath, dtype, [d1, d2])
                        else:
                            raw = utils._setup_data_ts(
                                filepath, dtype, [d1, d2, d3, d4, d5])
                            data = raw[['Timestamp', dtype]].copy()
                            acc = raw[['Timestamp', 'X', 'Y', 'Z']].copy()

                # Handle Empatica files
                elif file_type == 'E4':
                    ds = False  # disable downsampling
                    total_progress = 6
                    perc = (2 / total_progress) * 100
                    set_progress((perc * 100, f'{perc:.0f}%'))

                    E4 = heartview.Empatica(filepath)
                    e4_data = E4.preprocess()

                    # Accelerometer data
                    acc = e4_data.acc
                    acc.to_csv(str(temp_path / f'{file}_ACC.csv'),
                               index = False)

                    # EDA data
                    eda = e4_data.eda
                    eda.to_csv(str(temp_path / f'{file}_EDA.csv'),
                               index = False)

                    # BVP data
                    data = e4_data.bvp
                    fs = e4_data.bvp_fs

                # Preprocess any acceleration data
                try:
                    acc['Magnitude'] = ACC.compute_magnitude(
                        acc['X'], acc['Y'], acc['Z'])
                    acc.to_csv(str(temp_path / f'{file}_ACC.csv'),
                               index = False)
                except:
                    acc = None

                # Preprocess data
                preprocessed_data = utils._preprocess_cardiac(
                    data, dtype, fs, seg_size, beat_detector,
                    artifact_method, artifact_tol, filt_on, acc, ds,
                    set_progress)

                # Write preprocessed data to 'temp' folder
                preprocessed_data[0].to_csv(
                    str(temp_path / f'{file}_{dtype}.csv'), index = False)
                preprocessed_data[1].to_csv(
                    str(temp_path / f'{file}_IBI.csv'), index = False)
                preprocessed_data[2].to_csv(
                    str(temp_path / f'{file}_SQA.csv'), index = False)

                # Write downsampled data to '_render' folder
                if ds:
                    preprocessed_data[3].to_csv(
                        str(render_dir / 'signal.csv'), index = False)
                    preprocessed_data[4].to_csv(
                        str(render_dir / 'ibi.csv'), index = False)
                    if preprocessed_data[5] is not None:
                        preprocessed_data[5].to_csv(
                            str(render_dir / 'acc.csv'), index = False)
                    ds_fs = preprocessed_data[6]
                else:
                    # Write preprocessed data to '_render' folder
                    preprocessed_data[0].to_csv(
                        str(render_dir / 'signal.csv'), index = False)
                    preprocessed_data[1].to_csv(
                        str(render_dir / 'ibi.csv'), index = False)
                    acc.to_csv(
                        str(render_dir / 'acc.csv'), index = False)

            # Store data variables in memory
            preprocessed['file type'] = file_type
            preprocessed['data type'] = dtype
            preprocessed['fs'] = fs
            preprocessed['downsampled fs'] = ds_fs if ds else fs
            preprocessed['filename'] = filename

            perc = 100
            set_progress((perc * 100, f'{perc:.0f}%'))
            sleep(0.7)

            return dtype_error, map_error, preprocessed

    # == Create Beat Editor editing files =====================================
    @app.callback(
        [Output('be-create-file', 'children'),
         Output('beat-editor-spinner', 'children'),
         Output('open-beat-editor', 'disabled')],
        Input('subject-dropdown', 'options'),
        [State('memory-db', 'data'),
         State('toggle-filter', 'on')],
        prevent_initial_call = True
    )
    def create_beat_editor_files(all_subjects, memory, filt_on):
        """Create Beat Editor _edit.json files for uploaded files and
        enable the 'Beat Editor' button."""
        file_type = memory['file type']
        data_type = memory['data type']
        fs = memory['fs']
        signal_col = 'Filtered' if filt_on else data_type

        # Handle batch files
        if file_type == 'batch':
            filenames = sorted([s for s in all_subjects.values()])
            for name in filenames:
                data = pd.read_csv(temp_path / f'{name}_{data_type}.csv')
                ts_col = 'Timestamp' if 'Timestamp' in data.columns else None
                beats_ix = data[data.Beat == 1].index.values
                if 'Artifact' in data.columns:
                    artifacts_ix = data[data.Artifact == 1].index.values
                else:
                    artifacts_ix = None

                # Downsample Beat Editor data to match dashboard render
                ds, _, _, ds_fs = utils._downsample_data(
                    data, fs, signal_col, beats_ix, artifacts_ix)
                heartview.write_beat_editor_file(
                    ds, ds_fs, signal_col, 'Beat', ts_col, name,
                    batch = True, verbose = False)

        # Handle single files
        else:
            filename = Path(memory['filename']).stem
            data = pd.read_csv(str(temp_path / f'{filename}_{data_type}.csv'))
            ts_col = 'Timestamp' if 'Timestamp' in data.columns else None
            beats_ix = data[data.Beat == 1].index.values
            if 'Artifact' in data.columns:
                artifacts_ix = data[data.Artifact == 1].index.values
            else:
                artifacts_ix = None

            # Downsample Beat Editor data to match dashboard render
            ds, _, _, ds_fs = utils._downsample_data(
                data, fs, signal_col, beats_ix, artifacts_ix)
            heartview.write_beat_editor_file(
                ds, ds_fs, signal_col, 'Beat', ts_col, filename,
                verbose = False)

        # Beat Editor button icon
        btn_icon = html.I(className = 'fa-solid fa-arrow-up-right-from-square')

        return None, btn_icon, False

    # ===================== ARTIFACT IDENTIFICATION MODAL =====================
    @app.callback(
        Output('artifact-identification-modal', 'is_open'),
        [Input('artifact-method-help', 'n_clicks'),
         Input('close-artifact-method-info', 'n_clicks')],
        prevent_initial_call = True
    )
    def toggle_artifact_identification_help(n1, n2):
        clicked = ctx.triggered_id
        if clicked == 'artifact-method-help':
            return True
        elif clicked == 'close-artifact-method-info':
            return False

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

    # === Populate file select dropdown =======================================
    @app.callback(
        [Output('subject-dropdown', 'options'),
         Output('subject-dropdown', 'value'),
         Output('subject-dropdown', 'disabled')],
        Input('memory-db', 'data'),
        prevent_initial_call = True
    )
    def update_data_select_dropdown(memory):
        """Populate the data select dropdown with the names of uploaded
        files."""
        file_type = memory['file type']
        drop_disabled = True  # dropdown is disabled by default

        # Handle batch files
        if file_type == 'batch':
            filenames = sorted(
                [p.name for p in render_dir.iterdir() if (p.is_dir())])
            drop_options = {name: name for name in filenames}
            drop_value = filenames[0]
            drop_disabled = False

        # Handle single E4, Actiwave, and CSV files
        else:
            filename = Path(memory['filename']).stem
            drop_value = filename
            drop_options = {filename: filename}

        return drop_options, drop_value, drop_disabled

    # === Update SQA plots ====================================================
    @app.callback(
        [Output('sqa-plot', 'figure'),
         Output('offcanvas', 'is_open', allow_duplicate = True),
         Output('postprocess-data', 'disabled')],
        [Input('memory-db', 'data'),
         Input('qa-charts-dropdown', 'value'),
         Input('subject-dropdown', 'value')],
        prevent_initial_call = True
    )
    def update_sqa_plot(memory, sqa_view, selected_subject):
        """Update the SQA plot based on the selected view and enable the
        'Postprocess' button."""
        file = selected_subject
        sqa = pd.read_csv(str(temp_path / f'{file}_SQA.csv'))
        fs = int(memory['downsampled fs'])
        sqa_view == 'default'

        cardio_sqa = SQA.Cardio(fs)
        if sqa_view == 'missing':
            sqa_plot = cardio_sqa.plot_missing(
                sqa, title = file)
        elif sqa_view == 'artifact':
            sqa_plot = cardio_sqa.plot_artifact(
                sqa, title = file)
        else:
            sqa_plot = cardio_sqa.plot_missing(
                sqa, title = file)
        return sqa_plot, False, False

    # === Update SQA table ====================================================
    @app.callback(
        [Output('summary-table', 'children'),
         Output('segment-dropdown', 'options'),
         Output('export-summary', 'disabled'),
         Output('export-mode', 'options'),
         Output('postprocess-export-mode', 'options')],
        [Input('memory-db', 'data'),
         Input('subject-dropdown', 'value'),
         State('subject-dropdown', 'options')],
        prevent_initial_call = True
    )
    def update_sqa_table(memory, selected_subject, all_subjects):
        """Update the SQA summary table and export batch options."""
        file = selected_subject
        data_type = memory['data type']
        file_type = memory['file type']
        if any(x is None for x in (data_type, file_type, file)):
            raise PreventUpdate

        sqa = pd.read_csv(str(temp_path / f'{file}_SQA.csv'))
        segments = sqa['Segment'].tolist()

        # Signal quality metrics
        if data_type in ['ECG', 'PPG', 'BVP']:
            table, quality_summary = utils._cardiac_summary_table(sqa)
            fnames = sorted([s for s in all_subjects.values()])

            # Create quality_summary.txt file
            for file in fnames:
                with open(str(temp_path / f'{file}_quality_summary.txt'), 'w') as f:

                    # Add filename to the first line
                    f.write(f'File: {file}\n')

                    for label, value in quality_summary:
                        f.write(f'{label}: {value}\n')
        else:
            table = utils._blank_table()

        # Enable 'Batch' mode export
        if file_type == 'batch':
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
         Output('beat-correction-status', 'data'),
         Output('beat-correction', 'hidden'),
         Output('accept-corrections', 'hidden'),
         Output('reject-corrections', 'hidden'),
         Output('revert-corrections', 'hidden')],
        [Input('memory-db', 'data'),
         Input('segment-dropdown', 'value'),
         Input('subject-dropdown', 'value'),
         Input('prev-segment', 'n_clicks'),
         Input('next-segment', 'n_clicks'),
         Input('beat-correction', 'n_clicks'),
         Input('accept-corrections', 'n_clicks'),
         Input('reject-corrections', 'n_clicks'),
         Input('revert-corrections', 'n_clicks'),
         Input('be-edited-trigger', 'children')],
        [State('beat-correction-status', 'data'),
         State('seg-size', 'value'),
         State('toggle-filter', 'on'),
         State('segment-dropdown', 'options'),
         State('artifact-method', 'value'),
         State('artifact-tol', 'value')],
        prevent_initial_call = True
    )
    def update_signal_plots(memory, selected_segment, selected_subject, prev_n, next_n, 
                            beat_correction_n, accept_corrections_n, reject_corrections_n, revert_corrections_n, beats_edited,
                            beat_correction_status, segment_size, filt_on, segments, artifact_method, artifact_tol):
        """Update the raw data plot based on the selected segment view."""
        if memory is None:
            raise PreventUpdate
        else:
            data_type = memory['data type']
            file_type = memory['file type']
            file = selected_subject
            if file_type == 'batch':
                render_subdir = render_dir / selected_subject
            else:
                render_subdir = render_dir
            signal = pd.read_csv(str(render_subdir / 'signal.csv'))
            fs = int(memory['downsampled fs'])

            trig = ctx.triggered_id

            # Reset selected_segment to 1 when new data is loaded
            if trig == 'memory-db':
                selected_segment = 1

            prev_tt_open = False
            next_tt_open = False

            # Handle prev/next segment clicks
            if trig == 'beat-correction':
                beats_ix = signal.loc[signal['Beat'] == 1].index.tolist()
                sqa = SQA.Cardio(fs)
                beats_ix_corrected, corrected_ibis, original, corrected = sqa.correct_interval(beats_ix, seg_size=segment_size, print_estimated_hr=False)
                signal.loc[beats_ix_corrected, 'Corrected'] = 1
                signal.to_csv(str(render_dir / 'signal.csv'), index = False)
                ibi_corrected = heartview.compute_ibis(
                    signal, fs, beats_ix_corrected, 'Timestamp')
                ibi_corrected.to_csv(str(temp_path / f'{file}_IBI_corrected.csv'), index = False)
                beat_correction_status['status'] = 'suggested'
            # Accept corrections and update signal and ibi files
            elif trig == 'accept-corrections':
                beat_correction_status['status'] = 'accepted'
                # Update signal and ibi files to reflect accepted corrections
                ibi = pd.read_csv(str(temp_path / f'{file}_IBI_corrected.csv'))
                ibi.to_csv(str(render_dir / 'ibi.csv'), index = False)
                ibi_corrected = None
                signal = pd.read_csv(str(render_dir / 'signal.csv'))
                signal.loc[signal['Beat'] == 1, 'Original Beat'] = 1
                signal['Beat'] = None
                signal.loc[signal['Corrected'] == 1, 'Beat'] = 1
                # Update artifacts
                beats_ix = signal.loc[signal['Beat'] == 1].index.tolist()
                sqa = SQA.Cardio(fs)
                artifacts_ix = sqa.identify_artifacts(
                    beats_ix, method = artifact_method, tol = artifact_tol,
                    initial_hr = 'auto')
                signal['Artifact'] = None
                signal.loc[artifacts_ix, 'Artifact'] = 1
                signal.to_csv(str(render_dir / 'signal.csv'), index = False)
            # Reject corrections and reset beat correction status
            elif trig == 'reject-corrections':
                beat_correction_status['status'] = None
                ibi_corrected = None
            # Revert corrections and update signal and ibi files to original
            elif trig == 'revert-corrections':
                beat_correction_status['status'] = None
                ibi_corrected = None
                signal = pd.read_csv(str(render_dir / 'signal.csv'))
                signal['Beat'] = None
                signal.loc[signal['Original Beat'] == 1, 'Beat'] = 1
                beats_ix = signal.loc[signal['Beat'] == 1].index.tolist()
                sqa = SQA.Cardio(fs)
                artifacts_ix = sqa.identify_artifacts(
                    beats_ix, method = artifact_method, tol = artifact_tol,
                    initial_hr = 'auto')
                signal['Artifact'] = None
                signal.loc[artifacts_ix, 'Artifact'] = 1
                signal.to_csv(str(render_dir / 'signal.csv'), index = False)
                ibi = heartview.compute_ibis(
                    signal, fs, beats_ix, 'Timestamp')
                ibi.to_csv(str(render_dir / 'ibi.csv'), index = False)
            else:
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
                # If beat correction status is suggested, render the corrected ibis
                if beat_correction_status['status'] == 'suggested':
                    ibi_corrected = pd.read_csv(str(temp_path / f'{file}_IBI_corrected.csv'))
                else:
                    ibi_corrected = None

            # If cardiac data was run
            if data_type in ['ECG', 'PPG', 'BVP']:
                ibi = pd.read_csv(str(render_subdir / 'ibi.csv'))
                try:
                    acc = pd.read_csv(str(render_subdir / 'acc.csv'))
                except FileNotFoundError:
                    acc = None

                y_axis = 'Filtered' if filt_on else data_type
                x_axis = 'Timestamp' if 'Timestamp' in signal.columns else 'Sample'
                ts_col = 'Timestamp' if 'Timestamp' in signal.columns else None

                # Create the signal subplots with beat edits applied
                if beats_edited == selected_subject:
                    saved_dir = beat_editor_dir / 'saved'
                    data_dir = beat_editor_dir / 'data'

                    # Get for the subject's '_edit.json' file
                    edit_file = [p for p in (
                        data_dir / f'{selected_subject}_edit.json',
                        data_dir / 'batch' / f'{selected_subject}_edit.json'
                    ) if p.is_file()][0]
                    edits = pd.read_json(
                        str(saved_dir / f'{selected_subject}_edited.json'))
                    editor_data = pd.read_json(edit_file)

                    # Process beat edits
                    data_edited = heartview.process_beat_edits(
                        editor_data, edits)
                    data_edited.to_csv(
                        str(temp_path / f'{selected_subject}_edited.csv'),
                        index = False)
                    data_edited_beats_ix = data_edited[
                        data_edited.Edited == 1].index.values

                    # Recompute IBIs with edited beats
                    ibi_edited = heartview.compute_ibis(
                        data_edited, fs, data_edited_beats_ix, ts_col)

                    # Render updated signal plots
                    signal_plots = heartview.plot_signal(
                        signal = data_edited, signal_type = data_type,
                        axes = (x_axis, 'Signal'), fs = fs,
                        peaks_map = {data_type: 'Edited'},
                        peaks_label = 'Edited Beat',
                        peaks_color = '#71b4eb',
                        edits_map = {data_type: {'Add': 'Added Beat',
                                                 'Unusable': 'Unusable'}},
                        acc = acc, ibi = ibi_edited,
                        seg_number = selected_segment,
                        seg_size = segment_size,
                    )

                else:
                    overlay_corrected = beat_correction_status['status'] == 'suggested'
                    correction_map = {data_type: 'Corrected'} if overlay_corrected else None
                    # Create the signal subplots for uploaded data
                    signal_plots = heartview.plot_signal(
                        signal = signal, signal_type = data_type,
                        axes = (x_axis, y_axis), fs = fs,
                        peaks_map = {data_type: 'Beat'},
                        artifacts_map = {data_type: 'Artifact'},
                        correction_map = correction_map,
                        acc = acc, ibi = ibi,
                        ibi_corrected = ibi_corrected,
                        seg_number = selected_segment,
                        seg_size = segment_size)

            # If EDA data was run, pass for now
            else:
                signal_plots = utils._blank_fig()

            beat_correction_hidden = beat_correction_status['status'] == 'suggested' or beat_correction_status['status'] == 'accepted'
            accept_corrections_hidden = beat_correction_status['status'] != 'suggested'
            reject_corrections_hidden = beat_correction_status['status'] != 'suggested'
            revert_corrections_hidden = beat_correction_status['status'] != 'accepted'

            return signal_plots, selected_segment, prev_tt_open, next_tt_open, beat_correction_status, beat_correction_hidden, accept_corrections_hidden, reject_corrections_hidden, revert_corrections_hidden

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
            State('subject-dropdown', 'value'),
            State('subject-dropdown', 'options'),
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
                       selected_subject, all_subjects, memory):
        """Export the SQA summary file and confirm the export."""
        if ctx.triggered_id in ('close-export', 'close-export2'):
            set_progress((0, ''))
            return [False, None, True, False, True]
        else:
            data_type = memory['data type']
            if export_mode == 'Single':
                file = selected_subject
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
                else:  # if data_type == 'ECG'
                    files2export.extend([
                        temp_path / f'{file}_ECG.csv',
                        temp_path / f'{file}_IBI.csv',
                        temp_path / f'{file}_quality_summary.txt'])
                if (temp_path / f'{file}_ACC.csv').exists():
                    files2export.append(temp_path / f'{file}_ACC.csv')

            else:  # if export_mode == 'Batch'
                fnames = sorted([s for s in all_subjects.values()])
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
                    else:  # if data_type == 'ECG'
                        files2export.extend([
                            temp_path / f'{f}_ECG.csv',
                            temp_path / f'{f}_IBI.csv',
                            temp_path / f'{f}_quality_summary.txt'
                        ])
                    if (temp_path / f'{f}_ACC.csv').exists():
                        files2export.append(temp_path / f'{f}_ACC.csv')

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
                            output_xls = utils._make_excel(subj_files)

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
                    xls_workbook = utils._make_excel(
                        files2export, set_progress = set_progress)

                    # Save Excel file
                    export = send_bytes(
                        xls_workbook.getvalue(),
                        f'sqa_summary_{current_time}.xlsx'
                    )

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

    # ======================== BEAT EDITOR ELEMENTS ===========================
    # === Update Beat Editor button label and trigger on edit =================
    @app.callback(
        [Output('beat-editor-btn-label', 'children'),
         Output('open-beat-editor', 'style'),
         Output('be-edited-trigger', 'children')],
        [Input('ok-beat-edits', 'n_clicks'),
         Input('cancel-beat-edits', 'n_clicks'),
         Input('subject-dropdown', 'value'),
         State('beat-editor-modal', 'is_open'),
         State('subject-dropdown', 'value'),
         State('be-edited-trigger', 'children')],
        prevent_initial_call = True
    )
    def reflect_beat_edits(n_apply, n_cancel, subject_dropdown,
                           beat_editor_open, selected_subject,
                           prev_beats_edited):
        """Update the Beat Editor button label, style, and trigger state
        when edits are detected in the saved file."""
        trig = ctx.triggered_id

        # Default button styling
        btn_label = 'Beat Editor'
        btn_style = {}
        beats_edited = None
        saved_dir = Path('beat-editor/saved')
        edited_file = saved_dir / f'{selected_subject}_edited.json'

        if trig == 'subject-dropdown':
            if prev_beats_edited != selected_subject and edited_file.exists():
                btn_label = 'Beats Edited'
                btn_style = {'background': '#f1ab2a'}
                beats_edited = selected_subject
        elif trig == 'ok-beat-edits':
            if edited_file.exists() and utils._check_beat_editor_status():
                btn_label = 'Beats Edited'
                btn_style = {'background': '#f1ab2a'}
                beats_edited = selected_subject
        elif trig == 'cancel-beat-edits':
            # Keep beats_edited only if it was already set previously
            if prev_beats_edited == selected_subject:
                beats_edited = selected_subject
                btn_label = 'Beats Edited'
                btn_style = {'background': '#f1ab2a'}

        return btn_label, btn_style, beats_edited

    # === Open/Close Beat Editor modal ========================================
    @app.callback(
        [Output('beat-editor-modal', 'is_open'),
         Output('beat-editor-content', 'children'),
         Output('ok-beat-edits', 'disabled')],
        [Input('open-beat-editor', 'n_clicks'),
         Input('ok-beat-edits', 'n_clicks'),
         Input('cancel-beat-edits', 'n_clicks'),
         State('subject-dropdown', 'value')],
        prevent_initial_call = True
    )
    def toggle_beat_editor(beat_editor_clicked, apply_beats_clicked,
                           beat_editor_cancel_clicked, selected_subject):
        """Open or close the Beat Editor modal."""
        clicked = ctx.triggered_id

        if clicked == 'open-beat-editor':
            data_dir = Path('beat-editor/data')
            batch_dir = data_dir / 'batch'
            try:
                if batch_dir.exists():
                    # Move any current JSON file back to 'beat-editor/data/batch'
                    current_data = list(data_dir.glob('*_edit.json'))
                    for f in current_data:
                        if selected_subject and f.name == f'{selected_subject}_edit.json':
                            continue
                        dest = batch_dir / f.name
                        try:
                            dest.unlink()
                        except Exception:
                            pass
                        shutil.move(f, dest)

                    # Move selected file's '_edit.json' to 'beat-editor/data'
                    if selected_subject:
                        src = batch_dir / f'{selected_subject}_edit.json'
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
                    if utils._check_beat_editor_status():
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
                                        'https://heartview.readthedocs.io/en/latest/'
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
        elif clicked in ('cancel-beat-edits', 'ok-beat-edits'):
            return False, None, False

    # ======================= POSTPROCESESING MODAL ==========================
    # === Open/Close Postprocessing modal ====================================
    @app.callback(
        Output('postprocess-modal', 'is_open', allow_duplicate = True),
        [Input('postprocess-data', 'n_clicks'),
         Input('cancel-postprocess', 'n_clicks')],
        [State('subject-dropdown', 'value'),
         State('postprocess-modal', 'is_open')],
        prevent_initial_call = True
    )
    def toggle_postprocessing(postprocess_clicked, cancel_clicked,
                              selected_subject, is_open):
        clicked = ctx.triggered_id
        if clicked == 'postprocess-data':
            if selected_subject is None:
                return is_open
            return True
        if clicked == 'cancel-postprocess':
            return False if is_open else is_open
        return is_open

    # === Select all output types ============================================
    @app.callback(
        [Output('postprocess-options', 'value'),
         Output('select-all', 'children')],
        [Input('select-all', 'n_clicks'),
         Input('postprocess-options', 'value'),
         State('postprocess-options', 'value')],
        prevent_initial_call = True
    )
    def select_all_output_types(n, selected, current_selection):
        all_output_types = [
            {'label': 'Raw and Cleaned Signal', 'value': 'signal_data'},
            {'label': 'Interval Series', 'value': 'interval_data'},
            {'label': 'Derived Features', 'value': 'features'},
            {'label': 'Signal Quality Metrics', 'value': 'sqa'}
        ]
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
            State('subject-dropdown', 'value'),
            State('subject-dropdown', 'options'),
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
                         all_subjects, outputs, window_size, step_size,
                         export_mode, export_fmt):
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
        total_progress = len(subjects) * n_files + 2
        def _update_progress(units = 1):
            nonlocal progress_done
            progress_done += units
            perc = (progress_done / max(total_progress, 1)) * 100
            set_progress((perc, f'{perc:.0f}%'))
        _update_progress()

        # Helper function for postprocessing one subject's file(s)
        def _postprocess_one(s: str) -> list[Path]:
            """Postprocess data for a selected subject and aggregate
            user-selected output files into a list of Path objects.
            :param s: The selected subject.
            """
            out = []
            data = pd.read_csv(temp_path / f'{s}_{data_type}.csv')
            if data_type in ('ECG', 'BVP', 'PPG'):

                # -------------------- Process Edited Data ------------------
                edited_file = temp_path / f'{s}_edited.csv'
                has_edits = edited_file.exists()

                # Get sampling rate of Beat Editor data
                beat_editor_fs = int(memory['downsampled fs'])  # < ~250

                if has_edits:

                    # Get indices of beat edits
                    edited = pd.read_csv(str(edited_file))
                    edited_beats_ix = edited[edited.get(
                        'Edited').eq(1)].index.values

                    # Set default indices
                    deletions_ix = np.array([], dtype = int)
                    additions_ix = np.array([], dtype = int)
                    unusable_ix = np.array([], dtype = int)

                    # Map edited beats to original sampliung rate
                    mapped_edits_ix = utils._map_beat_edits(
                        edited_beats_ix, beat_editor_fs, fs_full)
                    if 'Deleted Beat' in edited.columns:
                        deletions_ix = edited[edited.get(
                            'Deleted Beat').eq(1)].index.values
                    mapped_deletions_ix = utils._map_beat_edits(
                        deletions_ix, beat_editor_fs, fs_full)
                    if 'Added Beat' in edited.columns:
                        additions_ix = edited[edited.get(
                            'Added Beat').eq(1)].index.values
                    mapped_additions_ix = utils._map_beat_edits(
                        additions_ix, beat_editor_fs, fs_full)
                    if 'Unusable' in edited.columns:
                        unusable_ix = edited[edited.get(
                            'Unusable').eq(1)].index.values
                    mapped_unusable_ix = utils._map_beat_edits(
                        unusable_ix, beat_editor_fs, fs_full)

                    # Record edited beats in original data
                    data.loc[mapped_edits_ix, 'Edited'] = 1
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
                            blocks = starts[:, None] + np.arange(
                                k, dtype = int)
                            blocks = blocks[blocks < len(data)]
                            data.loc[np.unique(blocks.ravel()), 'Unusable'] = 1
                        else:
                            starts = np.floor(
                                mapped_unusable_ix * ratio).astype(int)
                            ends = np.ceil(
                                (mapped_unusable_ix + 1) * ratio).astype(int)
                            ends = np.maximum(ends, starts + 1)
                            for s, e in zip(starts, ends):
                                parts = np.arange(
                                    s, min(e, len(data)), dtype = int)
                            if parts:
                                full = np.unique(np.concatenate(parts))
                                data.loc[full, 'Unusable'] = 1

                    # Rewrite to temp_path
                    data.to_csv(
                        str(temp_path / f'{s}_{data_type}_cleaned.csv'),
                        index = False)
                    (temp_path / f'{s}_{data_type}.csv').unlink()

                # ------------------- Raw and Cleaned Data -------------------
                if want_signal:
                    sig_files = [p for p in signal_files if s in str(p)]
                    for sig_path in sig_files:
                        if sig_path.stem.endswith(data_type):
                            cleaned = temp_path / f'{s}_{data_type}_cleaned.csv'
                            if cleaned.exists():
                                out.append(cleaned)
                            else:
                                out.append(sig_path)
                        else:
                            out.append(sig_path)

                # ------------ Interval Series / Derived Features ------------
                if want_int or want_feats:
                    ts_col = 'Timestamp' if 'Timestamp' in data.columns else \
                        'Sample'

                    # Recompute IBIs with edited beats
                    if has_edits:
                        edited_ibi = heartview.compute_ibis(
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

                    if want_int:
                        p = temp_path / f'{s}_IBI.csv'
                        out.append(p)

                    if want_feats:
                        ibi = pd.read_csv(str(temp_path / f'{s}_IBI.csv'))
                        ibi.set_index(ts_col, inplace = True)
                        hrv = get_hrv_features(
                            data = ibi.dropna()['IBI'],
                            window_length = window_size,
                            window_step_size = step_size,
                            domains = ['td', 'fd', 'nl', 'stat'],
                            threshold = 0.5, clean_data = False)

                        # Write HRV data to temp_path
                        p = temp_path / f'{s}_HRV.csv'
                        hrv.to_csv(str(p))
                        out.append(p)

                # ------------------ Signal Quality Metrics ------------------
                if want_sqa:
                    for p in [temp_path / f'{s}_SQA.csv',
                              temp_path / f'{s}_quality_summary.txt']:
                        if p.exists():
                            out.append(p)
            return out

        # Get all files for export
        if export_mode.lower() == 'single':
            files2make = _postprocess_one(selected_subject)
        elif export_mode.lower() == 'batch':
            files2make = []
            for s in subjects:
                files2make.append(_postprocess_one(s))
        _update_progress()  # add progress after file aggregation

        # Write data in requested format
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        buf = BytesIO()
        if export_fmt.lower() == 'excel':
            if export_mode.lower() == 'single':
                ext = 'xlsx'
                buf = utils._make_excel(
                    files2make, set_progress = set_progress,
                    progress_start = progress_done,
                    progress_total = total_progress)
            elif export_mode.lower() == 'batch':
                ext = 'zip'
                with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for subj in subjects:
                        files = [f for flist in files2make for f in flist
                                 if subj in str(f)]
                        xls_out = utils._make_excel(files)
                        zf.writestr(f'{subj}_processed.xlsx', xls_out.getvalue())
                        _update_progress()  # add progress per subject
        elif export_fmt.lower() == 'zip':  # if 'zip' format
            ext = 'zip'
            if export_mode.lower() == 'single':
                buf = utils._make_zip(
                    files2make, set_progress = set_progress,
                    progress_start = progress_done,
                    progress_total = total_progress)
            elif export_mode.lower() == 'batch':
                files = [f for sub in files2make for f in sub]
                buf = utils._make_zip(
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
        _update_progress()

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