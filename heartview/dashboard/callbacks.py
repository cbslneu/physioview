from dash import html, Input, Output, State, ctx, callback
from dash.exceptions import PreventUpdate
from dash.dcc import send_bytes
from heartview import heartview
from heartview.pipeline import ACC, SQA
from heartview.pipeline.EDA import compute_tonic_scl
from heartview.dashboard import utils
from flirt.hrv import get_hrv_features
from flirt.eda import get_eda_features
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
            Output('e4-data-type-container', 'hidden'),  # E4 data types div
            Output('memory-load', 'data'),
        ],
        id = 'dash-uploader'
    )
    def db_get_file_types(filenames):
        """Save the data type to the local memory depending on the file
        type."""
        if not filenames:
            # raise PreventUpdate
            return [], True, True, True, None

        session_path = Path(filenames[0]).parent
        latest_file = sorted(
            session_path.iterdir(), key = lambda f: -stat(f).st_mtime)[0]
        filename = str(latest_file)

        # Default visibility
        disable_run = True
        disable_configure = True
        hide_e4_dtypes = True

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
        configs = utils._load_config(cfg_file[0])
        return configs

    # ======================== ENABLE DATA PARAMETERS =========================
    @app.callback(
        [Output('resample', 'hidden'),
         Output('resampling-rate', 'disabled'),
         Output('load-temperature', 'hidden'),
         Output('temp-upload-section', 'hidden'),
         Output('cardio-preprocessing', 'hidden'),
         Output('eda-preprocessing', 'hidden', allow_duplicate = True),
         Output('scr-amplitude-threshold', 'hidden'),
         Output('beat-detectors', 'options'),
         Output('beat-detectors', 'value'),
         Output('seg-size', 'value', allow_duplicate = True)],
        [Input('e4-data-types', 'value'),
         Input('data-types', 'value'),
         Input('toggle-resample', 'on'),
         Input('toggle-scr-detection', 'on'),
         Input('toggle-temp-data', 'on'),
         State('memory-load', 'data')],
        prevent_initial_call = True
    )
    def db_enable_dtype_specific_parameters(e4_dtype, dtype, toggle_rs_on,
                                            toggle_scr_on, toggle_temp_on,
                                            loaded_data):
        """Enable parameters specific to data types of CSV sources."""
        load_temp_hidden = True
        temp_upload_hidden = True
        cardio_preprocess_hidden = True
        eda_preprocess_hidden = True
        scr_amp_thresh_hidden = True
        resample_hidden = True
        resample_disabled = True
        beat_detectors = []
        default_beat_detector = None
        data_source = loaded_data['source']
        seg_size = 60

        # Handle EDA components
        if dtype == 'EDA' or e4_dtype == 'EDA':
            resample_hidden = False
            load_temp_hidden = False
            eda_preprocess_hidden = False
            seg_size = 180
            if toggle_rs_on is True:
                resample_disabled = False
            if toggle_scr_on is True:
                scr_amp_thresh_hidden = False
            if toggle_temp_on is True:
                temp_upload_hidden = False

        # Handle cardiac components
        if dtype in ('PPG', 'ECG')  or e4_dtype == 'PPG' or data_source == 'Actiwave':
            cardio_preprocess_hidden = False
            if dtype == 'PPG' or e4_dtype == 'PPG':
                beat_detectors = [
                    {'label': 'Elgendi et al. (2013)', 'value': 'erma'},
                    {'label': 'Van Gent et al. (2018)', 'value': 'adaptive_threshold'}]
                default_beat_detector = 'adaptive_threshold'
            else:
                beat_detectors = [
                    {'label': 'Manikandan & Soman (2012)', 'value': 'manikandan'},
                    {'label': 'Engels & Zeelenberg (1979)', 'value': 'engzee'},
                    {'label': 'Nabian et al. (2018)', 'value': 'nabian'},
                    {'label': 'Pan & Tompkins (1985)', 'value': 'pantompkins'}
                ]
                default_beat_detector = 'manikandan'

        return [resample_hidden, resample_disabled,
                load_temp_hidden, temp_upload_hidden,
                cardio_preprocess_hidden, eda_preprocess_hidden,
                scr_amp_thresh_hidden, beat_detectors, default_beat_detector,
                seg_size]

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

        temperature_data = utils._parse_temp_csv(contents)
        if temperature_data.shape[1] != 1:
            file_check = [html.I(className = 'fa-solid fa-circle-xmark'),
                          html.Span('Invalid data type!')]

        col = temperature_data.iloc[:, 0]
        if pd.api.types.is_string_dtype(col) and not col.str.replace(
                '.', '').str.isnumeric().all():
            col = col.iloc[1:]
        temp_vals = pd.to_numeric(col, errors = 'coerce').dropna().tolist()
        data['TEMP'] = temp_vals

        # Update uploader text to show the filename
        if filename:
            uploaded = html.Span(f'{filename}')
        else:
            uploaded = html.Span('Select File...')

        return data, file_check, uploaded

    # === Clear uploaded temperature file =====================================
    @app.callback(
        [Output('temp-uploader', 'contents'),
         Output('temp-uploader', 'filename'),
         Output('temp-uploader', 'last_modified'),
         Output('temp-uploader', 'children', allow_duplicate = True)],
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
            return None, None, None, 'Select File...'
        if trig == 'duplicate-temp-error-modal' and not error_is_open:
            return None, None, None, 'Select File...'
        raise PreventUpdate

    # =================== POPULATE PARAMETERIZATION FIELDS ====================
    @app.callback(
        [Output('setup-data', 'hidden'),
         Output('preprocess-data', 'hidden'),
         Output('eda-preprocessing', 'hidden', allow_duplicate = True),
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
         Output('sampling-rate', 'value'),
         Output('seg-size', 'value', allow_duplicate = True),
         Output('artifact-method', 'value'),
         Output('artifact-tol', 'value'),
         Output('toggle-filter', 'on'),
         Output('toggle-scr-detection', 'on'),
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
        hide_setup = False
        hide_preprocess = False
        hide_eda_preprocess = True
        hide_segsize = False
        hide_data_types = False
        hide_data_vars = False
        hide_variable_error = True

        # Default toggler states
        temp_on = False
        filter_on = False
        scr_on = False

        # Default parameter values
        base_headers = ['<Var>', '<Var>']
        drop_values = [base_headers[:] for _ in range(6)]
        temp_uploader_disabled = False
        temp_uploader_text = 'Select File...'
        temp_options = []
        temp_value = None
        artifact_method = 'cbd'
        artifact_tol = 1
        seg_size = 60
        fs = 500
        dtype = None
        eda_min = 0.2
        eda_max = 40

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
                hide_eda_preprocess = False
                hide_data_types = True
                hide_data_vars = True
                fs = 64
                seg_size = 180
                if toggle_config_on:
                    seg_size = configs['segment size']
                    fs = configs['sampling rate']
                    dtype = configs['data type']
                    scr_on = configs['scr detection']

            # -- csv sources -------------------------------------------------
            elif memory['source'] == 'csv':
                if toggle_config_on:
                    pass
                else:
                    headers = utils._get_csv_headers(memory['filename'])
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
            scr_on = configs['scr detection']
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
                drop_values = [h for h in configs['headers'].values()
                               if h is not None]
                # drop_values = [[h for h in headers if h is not None] for _
                #                in range(6)]

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
            hide_setup, hide_preprocess, hide_eda_preprocess, hide_segsize,
            hide_data_types, dtype, hide_data_vars, hide_variable_error,

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

            fs, seg_size, artifact_method, artifact_tol, filter_on, scr_on,
            eda_min, eda_max
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
         State('toggle-scr-detection', 'on'),
         State('scr-amp-thresh', 'value'),
         State('eda-valid-min', 'value'),
         State('eda-valid-max', 'value'),
         State('config-filename', 'value')],
        prevent_initial_call = True
    )
    def write_confirm_config(n, data, dtype, fs, d1, d2, d3, d4, d5,
                             seg_size, artifact_method, artifact_tol,
                             filter_on, temp_on, temp_var, scr_on,
                             scr_amp, eda_min, eda_max, filename):
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
                filter_on, scr_on, scr_amp, headers, temp_on, temp_var,
                eda_min, eda_max)
            download = {'content': json_object, 'filename': f'{filename}.json'}
            return [download, 1]

    # ============================= RUN PIPELINE ==============================
    @callback(
        output = [
            Output('dtype-validator', 'is_open'),
            Output('mapping-validator', 'is_open'),
            Output('pipeline-error-modal', 'is_open'),
            Output('pipeline-error-message', 'children'),
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
            State('temperature-load', 'data'),
            State('temp-variable', 'value'),
            State('beat-detectors', 'value'),
            State('seg-size', 'value'),
            State('artifact-method', 'value'),
            State('artifact-tol', 'value'),
            State('toggle-filter', 'on'),
            State('toggle-scr-detection', 'on'),
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
    def run_pipeline(set_progress, n, load_data, e4_dtype, dtype, fs, rs, d1,
                     d2, d3, d4, d5, temp_data, temp_var, beat_detector,
                     seg_size, artifact_method, artifact_tol, filt_on, scr_on,
                     scr_amp, eda_min, eda_max):
        """Read Actiwave Cardio, Empatica E4, or CSV-formatted data, save
        the data to the local memory, and load the progress spinner."""

        dtype_error = False
        map_error = False
        pipeline_error = False
        temp_input_error = False

        if ctx.triggered_id == 'run-data':

            # Reset progress bar
            set_progress((0, '0%'))

            # Create '_render' folder
            render_dir.mkdir(parents = True, exist_ok = True)

            file_type = load_data['source']
            if file_type not in ('Actiwave', 'E4'):
                if dtype is None:
                    dtype_error = True
                    return dtype_error, map_error, pipeline_error, '', \
                        temp_input_error, None
                elif d2 is None:
                    map_error = True
                    return dtype_error, map_error, pipeline_error, '', \
                        temp_input_error, None

            filepath = load_data['filename']
            filename = Path(filepath).name  # e.g., "example.csv"
            file = Path(filepath).stem

            # Set up storage
            memory = {}

            # Enable downsampling if fs or rs is greater than the sampling
            # rate (~250 Hz) of the render data
            ds = fs > 250 and (rs is None or rs > 250)
            ds_data, ds_ibi, ds_acc, ds_fs = None, None, None, None

            # Initialize for uploads without IBI or ACC
            ibi, acc = None, None

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

                    # If timestamps are given
                    if d1 is not None:
                        has_ts = True
                        # No acceleration data
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data, acc = utils._setup_data(
                                f, dtype, [d1, d2], temp_var, has_ts)
                        # With acceleration data
                        else:
                            data, acc = utils._setup_data(
                                f, dtype, [d1, d2, d3, d4, d5], temp_var,
                                has_ts)
                    else:
                        has_ts = False
                        # No acceleration data
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data, acc = utils._setup_data(
                                f, dtype, [d2], temp_var, has_ts)
                        # With acceleration data
                        else:
                            data, acc = utils._setup_data(
                                f, dtype, [d2, d3, d4, d5], temp_var, has_ts)

                    # Preprocess any acceleration data
                    if acc is not None:
                        acc['Magnitude'] = ACC.compute_magnitude(
                            acc['X'], acc['Y'], acc['Z'])
                        if has_ts:
                            unix_fmt = utils._check_unix(acc.Timestamp)
                            if unix_fmt is not None:
                                acc.Timestamp = pd.to_datetime(
                                    acc.Timestamp, unit = unix_fmt)
                        acc.to_csv(
                            str(temp_path / f'{fname}_ACC.csv'),
                            index = False)

                    # ---- cardiac data --------------------------------------
                    if dtype in ('ECG', 'PPG'):
                        try:
                            preprocessed = utils._preprocess_cardiac(
                                data, dtype, fs, seg_size, beat_detector,
                                artifact_method, artifact_tol, filt_on,
                                acc_data = acc, downsample = ds)
                        except Exception as e:
                            pipeline_error = True
                            error_type = type(e).__name__
                            error_msg = f'{error_type}: {e}'
                            return dtype_error, map_error, pipeline_error, \
                                error_msg, temp_input_error, None
                        metrics = preprocessed[2]

                        # Write IBI data to 'temp' folder
                        ibi = preprocessed[1]
                        ibi.to_csv(
                            str(temp_path / f'{fname}_IBI.csv'), index = False)

                        # Get downsampled data for rendering
                        if len(preprocessed) > 3:
                            ds_data = preprocessed[3]
                            ds_ibi = preprocessed[4]
                            ds_acc = preprocessed[5]
                            ds_fs = preprocessed[6]

                    # ---- EDA data ------------------------------------------
                    else:
                        temp = data['TEMP'].values if 'TEMP' in data.columns \
                            else None
                        try:
                            preprocessed = utils._preprocess_eda(
                                data, fs, rs, temp, seg_size, filt_on, scr_on,
                                scr_amp, eda_min, eda_max)
                        except Exception as e:
                            pipeline_error = True
                            error_type = type(e).__name__
                            error_msg = f'{error_type}: {e}'
                            return dtype_error, map_error, pipeline_error, \
                                error_msg, temp_input_error, None
                        metrics = preprocessed[1]

                        # Get downsampled data for rendering
                        if len(preprocessed) > 2:
                            ds_data = preprocessed[2]
                            ds_acc = preprocessed[3]
                            ds_fs = preprocessed[4]

                    # Write preprocessed data and metrics to 'temp' folder
                    preprocessed_data = preprocessed[0]
                    preprocessed_data.to_csv(
                        str(temp_path / f'{fname}_{dtype}.csv'), index = False)
                    metrics.to_csv(
                        str(temp_path / f'{fname}_SQA.csv'), index = False)

                    # Write any downsampled data to '_render' folder
                    render_subdir = render_dir / fname
                    render_subdir.mkdir(parents = True, exist_ok = True)
                    if ds_data is not None:
                        ds_data.to_csv(
                            str(render_subdir / 'signal.csv'), index = False)
                    else:
                        preprocessed_data.to_csv(
                            str(render_subdir / 'signal.csv'), index = False)
                    if ibi is not None:
                        if ds_ibi is not None:
                            ds_ibi.to_csv(
                                str(render_subdir / 'ibi.csv'), index = False)
                        else:
                            ibi.to_csv(
                                str(render_subdir / 'ibi.csv'), index = False)
                    if ds_acc is not None:
                        ds_acc.to_csv(
                            str(render_subdir / 'acc.csv'), index = False)
                    else:
                        if acc is not None:
                            acc.to_csv(
                                str(render_subdir / 'acc.csv'), index = False)

                    perc = ((idx + 2) / total_progress) * 100
                    set_progress((perc, f'{perc:.0f}%'))
                    sleep(0.5)

            else:
                # Update progress bar for all single-file sources: 33%
                total_progress = 6
                perc = (2 / total_progress) * 100
                set_progress((perc, f'{perc:.0f}%'))
                sleep(0.5)

                # -- Actiwave Cardio sources ---------------------------------
                if file_type == 'Actiwave':
                    dtype = 'ECG'

                    # Prepare Actiwave Cardio data
                    actiwave = heartview.Actiwave(filepath)
                    actiwave_data = actiwave.preprocess(time_aligned = True)
                    data = actiwave_data[['Timestamp', dtype]].copy()
                    acc = actiwave_data[['Timestamp', 'X', 'Y', 'Z']].copy()
                    acc.to_csv(
                        str(temp_path / f'{file}_ACC.csv'), index = False)
                    fs = actiwave.get_ecg_fs()

                # -- Empatica E4 sources -------------------------------------
                elif file_type == 'E4':
                    E4 = heartview.Empatica(filepath)
                    e4_data = E4.preprocess()

                    # Accelerometer data
                    acc = e4_data.acc
                    acc.to_csv(
                        str(temp_path / f'{file}_ACC.csv'), index = False)

                    # Extract and save EDA data
                    if e4_dtype == 'EDA':
                        dtype = 'EDA'
                        eda = e4_data.eda
                        eda.to_csv(
                            str(temp_path / f'{file}_EDA.csv'), index = False)
                        fs = e4_data.eda_fs
                        data = eda.copy()

                        # Extract accompanying skin temperature data
                        temp = e4_data.temp
                        temp.to_csv(
                            str(temp_path / f'{file}_TEMP.csv'), index = False)

                    # Extract and save BVP data
                    elif e4_dtype == 'PPG':
                        dtype = 'BVP'
                        bvp = e4_data.bvp
                        bvp.to_csv(
                            str(temp_path / f'{file}_BVP.csv'), index = False)
                        fs = e4_data.bvp_fs
                        data = bvp.copy()

                # -- csv sources ---------------------------------------------
                else:

                    # Check if duplicate temperature inputs
                    if temp_data is not None and temp_var is not None:
                        temp_input_error = True
                        return dtype_error, map_error, pipeline_error, '', \
                            temp_input_error, None

                    # If timestamps are given
                    if d1 is not None:
                        has_ts = True
                        # No acceleration data
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data, acc = utils._setup_data(
                                filepath, dtype, [d1, d2], temp_var, has_ts)
                        # With acceleration data
                        else:
                            data, acc = utils._setup_data(
                                filepath, dtype, [d1, d2, d3, d4, d5],
                                temp_var, has_ts)
                    else:
                        has_ts = False
                        # No acceleration data
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            data, acc = utils._setup_data(
                                filepath, dtype, [d2], temp_var, has_ts)
                        # With acceleration data
                        else:
                            data, acc = utils._setup_data(
                                filepath, dtype, [d2, d3, d4, d5], temp_var,
                                has_ts)

                # Update progress bar: 50%
                perc = (3 / total_progress) * 100
                set_progress((perc, f'{perc:.0f}%'))
                sleep(0.5)

                # Preprocess any acceleration data
                if acc is not None and not acc.empty:
                    acc['Magnitude'] = ACC.compute_magnitude(
                        acc['X'], acc['Y'], acc['Z'])
                    if 'Timestamp' in acc.columns:
                        unix_fmt = utils._check_unix(acc.Timestamp)
                        if unix_fmt is not None:
                            acc.Timestamp = pd.to_datetime(
                                acc.Timestamp, unit = unix_fmt)
                    acc.to_csv(str(temp_path / f'{file}_ACC.csv'),
                               index = False)

                # Update progress bar: 67%
                perc = (4 / total_progress) * 100
                set_progress((perc, f'{perc:.0f}%'))
                sleep(0.5)

                # Preprocess any cardiac data
                if dtype in ('ECG', 'PPG', 'BVP') or e4_dtype == 'PPG':
                    try:
                        preprocessed = utils._preprocess_cardiac(
                            data, dtype, fs, seg_size, beat_detector,
                            artifact_method, artifact_tol, filt_on,
                            acc_data = acc, downsample = ds)
                    except Exception as e:
                        pipeline_error = True
                        error_type = type(e).__name__
                        error_msg = f'{error_type}: {e}'
                        return dtype_error, map_error, pipeline_error, \
                            error_msg, temp_input_error, None
                    metrics = preprocessed[2]

                    # Write IBI data to 'temp' folder
                    ibi = preprocessed[1]
                    ibi.to_csv(
                        str(temp_path / f'{file}_IBI.csv'), index = False)

                    # Get downsampled data for rendering
                    if len(preprocessed) > 3:
                        ds_data = preprocessed[3]
                        ds_ibi = preprocessed[4]
                        ds_acc = preprocessed[5]
                        ds_fs = preprocessed[6]

                # Preprocess any EDA data
                elif dtype == 'EDA' or e4_dtype == 'EDA':
                    if temp_data is not None:
                        temp = temp_data['TEMP']
                    elif 'TEMP' in data.columns:
                        temp = data['TEMP'].values
                    elif (temp_path / f'{file}_TEMP.csv').exists():
                        temperature = pd.read_csv(str(temp_path / f'{file}_TEMP.csv'))
                        temperature.Timestamp = pd.to_datetime(temperature.Timestamp)
                        data = pd.merge(data, temperature, on = 'Timestamp',
                                        how = 'inner')
                        temp = data['TEMP'].values
                    else:
                        temp = None

                    try:
                        preprocessed = utils._preprocess_eda(
                            data, fs, rs, temp, seg_size, filt_on,
                            scr_on, scr_amp, eda_min, eda_max)
                    except Exception as e:
                        pipeline_error = True
                        error_type = type(e).__name__
                        error_msg = f'{error_type}: {e}'
                        return dtype_error, map_error, pipeline_error, \
                            error_msg, temp_input_error, None
                    metrics = preprocessed[1]

                # Write preprocessed data and metrics to 'temp' folder
                preprocessed_data = preprocessed[0]
                preprocessed_data.to_csv(
                    str(temp_path / f'{file}_{dtype}.csv'), index = False)
                metrics.to_csv(
                    str(temp_path / f'{file}_SQA.csv'), index = False)

                # Write any downsampled data to '_render' folder
                render_subdir = render_dir / file
                render_subdir.mkdir(parents = True, exist_ok = True)
                if ds_data is not None:
                    ds_data.to_csv(
                        str(render_subdir / 'signal.csv'), index = False)
                else:
                    preprocessed_data.to_csv(
                        str(render_subdir / 'signal.csv'), index = False)

                # Write any IBI render data
                if ibi is not None:
                    if ds_ibi is not None:
                        ds_ibi.to_csv(
                            str(render_subdir / 'ibi.csv'), index = False)
                    else:
                        ibi.to_csv(
                            str(render_subdir / 'ibi.csv'), index = False)

                # Write any ACC render data
                if acc is not None and not acc.empty:
                    if ds_acc is not None:
                        ds_acc.to_csv(
                            str(render_subdir / 'acc.csv'), index = False)
                    else:
                        acc.to_csv(
                            str(render_subdir / 'acc.csv'), index = False)

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

            # Update progress bar: 100%
            set_progress((100, '100%'))
            sleep(1)

            return [dtype_error, map_error, pipeline_error, '',
                    temp_input_error, memory]

    # == Recompute SQA metrics for re-rendering ==============================
    @app.callback(
        Output('re-render-sqa-flag', 'data'),
        [Input('beat-correction-status', 'data'),
         Input('be-edited-trigger', 'children')],
        [State('memory-db', 'data'),
         State('subject-dropdown', 'value'),
         State('seg-size', 'value')],
        prevent_initial_call = True
    )
    def recompute_sqa(beat_correction_status, beats_edited, memory,
                      selected_subject, segment_size):
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
        file = selected_subject

        preprocessed_data = pd.read_csv(
            temp_path / f'{selected_subject}_{data_type}.csv')

        # Get manual beat edits and recomputed artifacts
        edited_file = temp_path / f'{file}_edited.csv'
        if edited_file.exists():
            edited = pd.read_csv(edited_file)
            edited_beats_ix = edited[
                edited.Edited == 1].index.values
            edited_artifacts_ix = edited[
                edited.Artifact == 1].index.values

            # Map edited indices back to original sampling rate
            beats_ix = utils._map_beat_edits(
                edited_beats_ix, beat_editor_fs, fs)
            artifacts_ix = utils._map_beat_edits(
                edited_artifacts_ix, beat_editor_fs, fs)

            # Remove any existing 'Artifact' column to prevent stale values
            # from affecting recomputation in sqa.compute_metrics()
            if 'Artifact' in preprocessed_data.columns:
                del preprocessed_data['Artifact']

        # Get auto-corrected beats and recomputed artifacts
        else:
            beats_ix = preprocessed_data[
                preprocessed_data.Beat == 1].index.values
            artifacts_ix = preprocessed_data[
                preprocessed_data.Artifact == 1].index.values

        ts_col = 'Timestamp' if 'Timestamp' in preprocessed_data.columns else None
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
        [Input('subject-dropdown', 'options'),
         Input('subject-dropdown', 'value'),
         Input('beat-correction-status', 'data')],
        [State('memory-db', 'data'),
         State('toggle-filter', 'on'),
         State('be-edited-trigger', 'children')],
        prevent_initial_call = True
    )
    def create_beat_editor_files(all_subjects, selected_subject,
                                 beat_correction_status, memory, filt_on,
                                 prev_beats_edited):
        """Create Beat Editor _edit.json files for uploaded cardiac files and
        enable the 'Beat Editor' button."""
        if memory is None:
            return None, True

        file_type = memory['file type']
        data_type = memory['data type']
        beat_editor_btn_disabled = True
        trig = ctx.triggered_id

        # Beat Editor button icon
        btn_icon = html.I(className = 'fa-solid fa-arrow-up-right-from-square')

        # Default spinner animation
        spinner_animation = ''

        if data_type in ('ECG', 'PPG', 'BVP'):

            if prev_beats_edited != selected_subject:
                fs = memory['fs']
                signal_col = 'Filtered' if filt_on else data_type

                # Handle batch files
                if file_type == 'batch' and trig != 'beat-correction-status':
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
                        ds, _, _, _, ds_fs = utils._downsample_data(
                            data, fs, data_type, beats_ix, artifacts_ix)
                        heartview.write_beat_editor_file(
                            ds, ds_fs, signal_col, 'Beat', ts_col, name,
                            batch = True, verbose = False)

                # Handle single files
                else:
                    if file_type == 'batch':
                        filename = selected_subject
                        batch = True
                    else:
                        filename = Path(memory['filename']).stem
                        batch = False
                    data = pd.read_csv(str(temp_path / f'{filename}_{data_type}.csv'))
                    ts_col = 'Timestamp' if 'Timestamp' in data.columns else None
                    beats_ix = data[data.Beat == 1].index.values
                    if 'Artifact' in data.columns:
                        artifacts_ix = data[data.Artifact == 1].index.values
                    else:
                        artifacts_ix = None

                    # Downsample Beat Editor data to match dashboard render
                    ds, _, _, _, ds_fs = utils._downsample_data(
                        data, fs, data_type, beats_ix, artifacts_ix)
                    heartview.write_beat_editor_file(
                        ds, ds_fs, signal_col, 'Beat', ts_col, filename,
                        batch = batch, verbose = False)
            else:
                spinner_animation = 'no-spin'

            beat_editor_btn_disabled = False

        return btn_icon, spinner_animation, beat_editor_btn_disabled

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
        [Output('subject-dropdown', 'options'),
         Output('subject-dropdown', 'value'),
         Output('subject-dropdown', 'disabled'),
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
        subject_drop_disabled = True  # dropdown is disabled by default
        sqa_drop_options = []  # empty SQA chart dropdown by default
        sqa_drop_value = ''

        # Handle batch files
        if file_type == 'batch':
            filenames = sorted(
                [p.name for p in render_dir.iterdir() if (p.is_dir())])
            drop_options = {name: name for name in filenames}
            drop_value = filenames[0]
            subject_drop_disabled = False

        # Handle single E4, Actiwave, and CSV files
        else:
            filename = Path(memory['filename']).stem
            drop_value = filename
            drop_options = {filename: filename}

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

        return drop_options, drop_value, subject_drop_disabled, \
            sqa_drop_options, sqa_drop_value

    # === Update SQA plots ====================================================
    @app.callback(
        [Output('sqa-plot', 'figure'),
         Output('offcanvas', 'is_open', allow_duplicate = True),
         Output('postprocess-data', 'disabled')],
        [Input('memory-db', 'data'),
         Input('qa-charts-dropdown', 'value'),
         Input('subject-dropdown', 'value'),
         Input('re-render-sqa-flag', 'data')],
        prevent_initial_call = True
    )
    def update_sqa_plot(memory, sqa_view, selected_subject, re_render_sqa_flag):
        """Update the SQA plot based on the selected view and enable the
        'Postprocess' button."""
        file = selected_subject
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
         Input('subject-dropdown', 'value'),
         Input('re-render-sqa-flag', 'data')],
        [State('subject-dropdown', 'options'),
         State('toggle-filter', 'on'),
         State('seg-size', 'value')],
        prevent_initial_call = True
    )
    def update_sqa_table(memory, selected_subject, re_render_sqa_flag,
                         all_subjects, filt_on, seg_size):
        """Update the SQA summary table and export batch options."""
        file = selected_subject
        data_type = memory['data type']
        file_type = memory['file type']
        if any(x is None for x in (data_type, file_type, file)):
            raise PreventUpdate

        sqa = pd.read_csv(str(temp_path / f'{file}_SQA.csv'))
        segments = sqa['Segment'].tolist()

        # Output signal quality table for cardiac data
        if data_type in ('ECG', 'PPG', 'BVP'):
            table, quality_summary = utils._cardiac_summary_table(sqa)

        # Output signal quality table for EDA data
        else:
            eda = pd.read_csv(temp_path / f'{file}_EDA.csv')
            signal_col = 'Filtered' if filt_on else 'EDA'
            eda_signal = eda[signal_col].to_numpy()
            tonic_scl = compute_tonic_scl(eda_signal)
            scr_series = eda['SCR'].values if 'SCR' in eda.columns else None
            table, quality_summary = utils._eda_summary_table(
                sqa, tonic_scl, scr_series, seg_size)

        # Create quality_summary.txt file(s)
        fnames = sorted([s for s in all_subjects.values()])
        for file in fnames:
            with open(str(temp_path / f'{file}_quality_summary.txt'), 'w') as f:

                # Add filename to the first line
                f.write(f'File: {file}\n')

                for label, value in quality_summary:
                    f.write(f'{label}: {value}\n')

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
         Output('open-beat-editor', 'disabled', allow_duplicate = True),
         Output('beat-correction-status', 'data'),
         Output('beat-correction', 'hidden'),
         Output('accept-corrections', 'hidden'),
         Output('reject-corrections', 'hidden'),
         Output('revert-corrections', 'hidden'),
         Output('plot-displayed', 'data')],
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
        [State('subject-dropdown', 'options'),
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
                            prev_n, next_n, beat_correction_n,
                            accept_corrections_n, reject_corrections_n,
                            revert_corrections_n, beats_edited, all_subjects,
                            beat_correction_status, segment_size, filt_on,
                            segments, artifact_method, artifact_tol,
                            temp_data, eda_min):
        """Update the raw data plot based on the selected segment view."""
        if memory is None:
            raise PreventUpdate
        else:
            data_type = memory['data type']
            file_type = memory['file type']
            fs = int(memory['downsampled fs'])
            file = selected_subject

            # Get render data for primary signal
            render_subdir = render_dir / selected_subject
            signal = pd.read_csv(str(render_subdir / 'signal.csv'))
            y_axis = 'Filtered' if filt_on else data_type
            x_axis = 'Timestamp' if 'Timestamp' in signal.columns else 'Sample'
            ts_col = 'Timestamp' if 'Timestamp' in signal.columns else None

            # Get ACC data if available
            try:
                acc = pd.read_csv(str(render_subdir / 'acc.csv'))
            except FileNotFoundError:
                acc = None

            trig = ctx.triggered_id

            # Reset selected_segment to 1 when new data is loaded
            if trig == 'memory-db':
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
                    for subject in all_subjects:
                        beat_correction_status[subject] = None

                def _save_temp_and_render(signal, file, data_type, fs, beats_ix,
                                          artifacts_ix, corrected_beats_ix = None):
                    beats_ix = np.array(beats_ix)
                    artifacts_ix = np.array(artifacts_ix)
                    if corrected_beats_ix is not None:
                        corrected_beats_ix = np.array(corrected_beats_ix)
                    signal.to_csv(str(temp_path / f'{file}_{data_type}.csv'), index = False)
                    ds_signal, ds_ibi, ds_ibi_corrected, _, _ = utils._downsample_data(
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
                    signal, beats_ix_corrected, ibi_corrected = utils._correct_beats(
                        signal, fs_full, beats_ix)
                    ibi_corrected.to_csv(str(temp_path / f'{file}_IBI_corrected.csv'), index = False)
                    signal, _, ibi_corrected = _save_temp_and_render(
                        signal, file, data_type, fs_full, beats_ix, artifacts_ix,
                        beats_ix_corrected)
                    ibi_corrected.to_csv(str(render_subdir / 'ibi_corrected.csv'), index = False)
                    beat_correction_status[selected_subject] = 'suggested'

                # Accept corrections and update signal and ibi files
                elif trig == 'accept-corrections':
                    beat_correction_status[selected_subject] = 'accepted'

                    # Update signal and ibi files to reflect accepted corrections
                    ibi = pd.read_csv(str(temp_path / f'{file}_IBI_corrected.csv'))
                    ibi.to_csv(str(temp_path / f'{file}_IBI.csv'), index = False)
                    ibi = pd.read_csv(str(render_subdir / 'ibi_corrected.csv'))
                    ibi.to_csv(str(render_subdir / 'ibi.csv'), index = False)
                    # ibi_corrected = None
                    signal = pd.read_csv(str(temp_path / f'{file}_{data_type}.csv'))
                    signal, beats_ix, artifacts_ix = utils._accept_beat_corrections(
                        signal, fs_full, artifact_method, artifact_tol)
                    signal, _, _ = _save_temp_and_render(
                        signal, file, data_type, fs_full, beats_ix, artifacts_ix)

                # Reject corrections and reset beat correction status
                elif trig == 'reject-corrections':
                    beat_correction_status[selected_subject] = None
                    # ibi_corrected = None

                # Revert corrections and update signal and ibi files to original
                elif trig == 'revert-corrections':
                    beat_correction_status[selected_subject] = None
                    # ibi_corrected = None
                    signal = pd.read_csv(str(temp_path / f'{file}_{data_type}.csv'))
                    signal, beats_ix, artifacts_ix = utils._revert_beat_corrections(
                        signal, fs_full, artifact_method, artifact_tol)
                    ibi = heartview.compute_ibis(
                        signal, fs_full, beats_ix, 'Timestamp')
                    ibi.to_csv(str(temp_path / f'{file}_IBI.csv'), index = False)
                    signal, ibi, _ = _save_temp_and_render(
                        signal, file, data_type, fs_full, beats_ix, artifacts_ix)
                    ibi.to_csv(str(render_subdir / 'ibi.csv'), index = False)

                # If beat correction status is suggested, render the corrected IBIs
                if beat_correction_status[selected_subject] == 'suggested':
                    ibi_corrected = pd.read_csv(str(render_subdir / 'ibi_corrected.csv'))

                # Get IBI data for rendering
                ibi = pd.read_csv(str(render_subdir / 'ibi.csv'))

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
                    data_edited_beats_ix = data_edited[
                        data_edited.Edited == 1].index.values

                    # Recompute artifacts with edited beats
                    sqa = SQA.Cardio(fs)
                    artifacts_edited = sqa.identify_artifacts(
                        data_edited_beats_ix, method = artifact_method,
                        tol = artifact_tol)
                    if 'Artifact' in data_edited.columns:
                        del data_edited['Artifact']
                    data_edited.loc[artifacts_edited, 'Artifact'] = 1

                    # Save edited data
                    data_edited.to_csv(
                        str(temp_path / f'{selected_subject}_edited.csv'),
                        index = False)

                    # Recompute IBIs with edited beats for rendering
                    ibi_edited = heartview.compute_ibis(
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
                            ends = np.insert(unusable_ix[breaks], 0, unusable_ix[-1])

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

                    # Render updated signal plots
                    signal_plots = heartview.plot_signal(
                        signal = data_edited, signal_type = data_type,
                        axes = (x_axis, 'Signal'), fs = fs,
                        peaks_map = {data_type: 'Edited'},
                        peaks_label = 'Edited Beat',
                        peaks_color = '#71b4eb',
                        edits_map = {data_type: {'Add': 'Added Beat',
                                                 'Unusable': 'Unusable'}},
                        artifacts_map = {data_type: 'Artifact'},
                        acc = acc, ibi = ibi_edited,
                        seg_number = selected_segment,
                        seg_size = segment_size)

                else:
                    overlay_corrected = beat_correction_status[selected_subject] == 'suggested'
                    correction_map = {data_type: 'Corrected'} if overlay_corrected else None

                    # Create cardiac signal subplots
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

                beat_correction_hidden = beat_correction_status[selected_subject] == 'suggested' \
                    or beat_correction_status[selected_subject] == 'accepted'
                accept_corrections_hidden = beat_correction_status[selected_subject] != 'suggested'
                reject_corrections_hidden = beat_correction_status[selected_subject] != 'suggested'
                revert_corrections_hidden = beat_correction_status[selected_subject] != 'accepted'

            # Otherwise create the EDA signal subplots
            else:
                eda_subplots = {'EDA': ['Phasic', y_axis, 'Tonic']}
                signal_types = [data_type]

                # Add temperature to subplots if data was provided
                if temp_data is not None:
                    signal['TEMP'] = temp_data['TEMP']
                if temp_data is not None or 'TEMP' in signal.columns:
                    signal_types.append('TEMP')
                    eda_subplots['TEMP'] = 'TEMP'

                # Check whether SCRs were detected
                has_scr = 'SCR' in signal.columns

                # Create EDA subplots
                signal_plots = heartview.plot_signal(
                    signal = signal, signal_type = signal_types,
                    axes = (x_axis, eda_subplots),
                    fs = fs,
                    peaks_map = {data_type: 'SCR'} if has_scr else None,
                    hline = eda_min, hline_name = 'Min. Valid EDA',
                    acc = acc, seg_number = selected_segment,
                    seg_size = segment_size)
                for trace in signal_plots.data:
                    if trace.name == y_axis:
                        trace.line.color = 'lightgrey'
                    if trace.name == 'Tonic':
                        trace.line.dash = 'dash'

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
            file_type = memory['file type']
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

    # === Enable/Disable Beat Correction buttons ==============================
    @app.callback(
        [Output('beat-correction', 'disabled'),
         Output('revert-corrections', 'disabled')],
        [Input('plot-displayed', 'data'),
         Input('be-edited-trigger', 'children')],
        [State('subject-dropdown', 'value'),
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
                btn_label = 'Beats Edited'
                btn_style = {'background': '#f1ab2a'}
                beats_edited = selected_subject

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
        [Output('postprocess-modal', 'is_open', allow_duplicate = True),
         Output('postprocess-options', 'options')],
        [Input('postprocess-data', 'n_clicks'),
         Input('cancel-postprocess', 'n_clicks')],
        [State('subject-dropdown', 'value'),
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
            data = pd.read_csv(temp_path / f'{s}_{data_type}.csv')

            # Get timestamp/sample column
            ts_col = 'Timestamp' if 'Timestamp' in data.columns else \
                'Sample'

            # Rename auto corrected beats column
            if 'Original Beat' in data.columns:
                data.rename(columns = {'Beat': 'Auto Corrected'},
                            inplace = True)

            has_edits = False
            if data_type in ('ECG', 'BVP', 'PPG'):

                # ---------------- Process Edited Cardiac Data ---------------
                edited_file = temp_path / f'{s}_edited.csv'
                has_edits = edited_file.exists()

                # Get sampling rate of Beat Editor data
                beat_editor_fs = int(memory['downsampled fs'])  # ~250

                if has_edits:

                    # Get indices of beat edits
                    edited = pd.read_csv(str(edited_file))
                    edited_beats_ix = edited[edited.get(
                        'Edited').eq(1)].index.values

                    # Set default indices
                    deletions_ix = np.array([], dtype = int)
                    additions_ix = np.array([], dtype = int)
                    unusable_ix = np.array([], dtype = int)

                    # Map edited beats to original sampling rate
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
                            blocks = starts[:, None] + np.arange(k, dtype = int)
                            blocks = blocks[blocks < len(data)]
                            data.loc[np.unique(blocks.ravel()), 'Unusable'] = 1
                        else:
                            starts = np.floor(
                                mapped_unusable_ix * ratio).astype(int)
                            ends = np.ceil(
                                (mapped_unusable_ix + 1) * ratio).astype(int)
                            ends = np.maximum(ends, starts + 1)
                            for s, e in zip(starts, ends):
                                parts = np.arange(s, min(e, len(data)), dtype = int)
                            if parts:
                                full = np.unique(np.concatenate(parts))
                                data.loc[full, 'Unusable'] = 1

                    # Reposition columns for clarity
                    if 'Original Beat' in data.columns:
                        beats_col = 'Original Beat'
                    else:
                        beats_col = 'Beat'
                    col_order = ['Segment', ts_col, data_type, 'Filtered',
                                 beats_col, 'Artifact', 'Auto Corrected',
                                 'Deleted Beat',  'Added Beat', 'Edited']
                    order = [c for c in col_order if c in data.columns]
                    data = data[order]

                    # Rewrite to temp_path
                    data.to_csv(
                        str(temp_path / f'{s}_{data_type}_cleaned.csv'),
                        index = False)
                    (temp_path / f'{s}_{data_type}.csv').unlink()

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
                                    'Beat': 'Auto Corrected',
                                    'Original Beat': 'Beat'}, inplace = True)
                            col_order = ['Segment', ts_col, data_type,
                                         'Filtered', 'Beat', 'Artifact',
                                         'Auto Corrected', 'Edited']
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

                    # Rewrite corrected IBI data to temp_path if available
                    else:
                        corrected_ibi = pd.read_csv(
                            temp_path / f'{s}_IBI_corrected.csv')
                        corrected_ibi.to_csv(
                            str(temp_path / f'{s}_IBI.csv'), index = False)

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

                # EDA feature extraction
                elif data_type == 'EDA':
                    data.set_index(ts_col, inplace = True)
                    eda_features = get_eda_features(
                        data = data['EDA'],
                        window_length = window_size,
                        window_step_size = step_size,
                        data_frequency = fs_full
                    )

                    # Write EDA features to temp_path
                    p = temp_path / f'{s}_Features.csv'
                    eda_features.to_csv(str(p))
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