from dash import html, Input, Output, State, ctx, callback
from dash.exceptions import PreventUpdate
from dash.dcc import send_bytes
from heartview import heartview
from heartview.pipeline import ACC, ECG, PPG, SQA
from heartview.dashboard import utils
from os import stat, path
from pathlib import Path
from time import sleep
from io import BytesIO
import dash_uploader as du
import zipfile
import pandas as pd

def get_callbacks(app):
    """Attach callback functions to the dashboard app."""

    root = Path(__file__).resolve().parents[3]
    temp_path = root / 'temp'
    render_dir = temp_path / '_render'
    render_dir.mkdir(parents = True, exist_ok = True)

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
        session_path = Path(filenames[0]).parent
        latest_file = sorted(
            session_path.iterdir(), key = lambda f: -stat(f).st_mtime)[0]
        filename = str(latest_file)

        if filenames[0].endswith(('edf', 'EDF')):
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
                disable_run = True
                disable_configure = True
        else:
            if filenames[0].endswith('zip'):
                z = zipfile.ZipFile(filename)
                if 'BVP.csv' not in z.namelist():
                    data = 'invalid'
                    file_check = [
                        html.I(className = 'fa-solid fa-circle-xmark'),
                        html.Span('Invalid data type!')
                    ]
                    disable_run = True
                    disable_configure = True
                else:
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
            if filenames[0].endswith('csv'):
                file_check = [
                    html.I(className = 'fa-solid fa-circle-check',
                           style = {'color': '#63e6be', 'marginRight': '5px'}),
                    html.Span('Data loaded.')
                ]
                data = {'source': 'csv',
                        'filename': filename}
                disable_run = False
                disable_configure = False

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
         Input('toggle-resample', 'on')],
        prevent_initial_call = True
    )
    def db_enable_dtype_specific_parameters(dtype, toggle_on):
        """Enable parameters specific to data types."""
        cardio_preprocess_hidden = True
        resample_hidden = True
        resample_disabled = True
        beat_detectors = []
        default_bd = None
        if dtype == 'EDA':
            resample_hidden = False
            if toggle_on is True:
                resample_disabled = False
            else:
                resample_disabled = True
        if dtype in ('PPG', 'ECG'):
            cardio_preprocess_hidden = False
            if dtype == 'PPG':
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
    def db_handle_upload_params(data, configs, toggle_config_on):
        """Output parameterization fields according to uploaded data."""
        loaded = ctx.triggered_id
        if loaded is None or data == 'invalid':
            raise PreventUpdate

        hide_setup = False
        hide_preprocess = False
        hide_segsize = False
        hide_data_types = False
        hide_data_vars = False

        drop1, drop2, drop3, drop4, drop5 = ([] for _ in range(5))
        drop_options = ['<Var>', '<Var>']
        filter_on = False
        artifact_method = 'cbd'
        artifact_tol = 1
        seg_size = 60
        fs = 500
        dtype = None

        if loaded == 'memory-load':
            if data['source'] == 'Actiwave':
                hide_setup = True
                hide_data_types = True
                hide_data_vars = True
                if toggle_config_on:
                    seg_size = configs['segment size']
                    fs = configs['sampling rate']
                    dtype = configs['data type']
            elif data['source'] == 'E4':
                hide_setup = True
                hide_data_types = True
                hide_data_vars = True
                fs = 64
                if toggle_config_on:
                    seg_size = configs['segment size']
                    fs = configs['sampling rate']
                    dtype = configs['data type']
            elif data['source'] == 'csv':
                if toggle_config_on:
                    pass
                else:
                    drop_options = utils._get_csv_headers(data['filename'])
                    drop1 = utils._get_csv_headers(data['filename'])
                    drop2 = utils._get_csv_headers(data['filename'])
                    drop3 = utils._get_csv_headers(data['filename'])
                    drop4 = utils._get_csv_headers(data['filename'])
                    drop5 = utils._get_csv_headers(data['filename'])

        elif loaded == 'config-memory':
            device = configs['device']
            dtype = configs['data type']
            seg_size = configs['segment size']
            fs = configs['sampling rate']
            if device == 'E4':
                hide_setup = True
                drop_options = []
                drop1 = None
                drop2 = None
                drop3 = None
                drop4 = None
                drop5 = None
            else:
                drop_options = list(configs['headers'].values())
                drop1 = drop_options[0]
                drop2 = drop_options[1]
                drop3 = drop_options[2]
                drop4 = drop_options[3]
                drop5 = drop_options[4]
            artifact_method = configs['artifact identification method']
            artifact_tol = configs['artifact tolerance']
            filter_on = configs['filters']

        if dtype == 'EDA':
            fs = 128

        return [hide_setup, hide_preprocess, hide_segsize, hide_data_types,
                dtype, hide_data_vars, drop_options, drop1, drop_options,
                drop2, drop_options, drop3, drop_options, drop4,
                drop_options, drop5, fs, seg_size, artifact_method,
                artifact_tol, filter_on]

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
                pass
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
            file_type = load_data['source']
            if file_type == 'E4':
                dtype = 'BVP'
            elif file_type == 'Actiwave':
                dtype = 'ECG'
            else:
                if dtype is None:
                    dtype_error = True
                    return dtype_error, map_error, None
                elif d1 is None or d2 is None:
                    map_error = True
                    return dtype_error, map_error, None

            total_progress = 6
            filepath = load_data['filename']
            filename = Path(filepath).name  # e.g., "example.csv"
            file = Path(filepath).stem
            data = {}
            perc = (1 / total_progress) * 100
            set_progress((perc, f'{perc:.0f}%'))

            # Handle Actiwave and ECG CSV files
            if file_type == 'Actiwave' or (
                    file_type == 'csv' and dtype == 'ECG'):
                if file_type == 'Actiwave':
                    actiwave = heartview.Actiwave(filepath)
                    ecg, acc = actiwave.preprocess()
                    acc.to_csv(str(temp_path / f'{file}_ACC.csv'), index = False)
                    fs = actiwave.get_ecg_fs()
                else:
                    # If no timestamps are provided
                    if d1 is None:
                        # If no acceleration data are provided
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            ecg = utils._setup_data_samples(
                                filepath, 'ECG', [d2])
                        else:
                            raw = utils._setup_data_samples(
                                filepath, 'ECG', [d2, d3, d4, d5])
                            ecg = raw[['Sample', 'ECG']].copy()
                            acc = raw[['Sample', 'X', 'Y', 'Z']].copy()
                    else:
                        if (d3 is None) & (d4 is None) & (d5 is None):
                            ecg = utils._setup_data_ts(
                                filepath, 'ECG', [d1, d2])
                        else:
                            raw = utils._setup_data_ts(
                                filepath, 'ECG', [d1, d2, d3, d4, d5])
                            ecg = raw[['Timestamp', 'ECG']].copy()
                            acc = raw[['Timestamp', 'X', 'Y', 'Z']].copy()

                    # Pre-process acceleration data
                    try:
                        acc['Magnitude'] = ACC.compute_magnitude(
                            acc['X'], acc['Y'], acc['Z'])
                        acc.to_csv(str(temp_path / f'{file}_ACC.csv'), index = False)
                    except:
                        acc = None

                # Filter ECG and detect beats
                perc = (2 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                if filt_on:
                    detect_beats = ECG.BeatDetectors(fs)
                    filt = ECG.Filters(fs)
                    ecg['Filtered'] = filt.filter_signal(ecg['ECG'])
                    perc = (3 / total_progress) * 100
                    set_progress((perc * 100, f'{perc:.0f}%'))
                    beats_ix = getattr(detect_beats, beat_detector)(
                        ecg['Filtered'])
                else:
                    detect_beats = ECG.BeatDetectors(fs, preprocessed = False)
                    beats_ix = getattr(detect_beats, beat_detector)(ecg['ECG'])
                ecg.loc[beats_ix, 'Beat'] = 1
                ecg.insert(0, 'Segment', ecg.index // (seg_size * fs) + 1)
                ecg.to_csv(str(temp_path / f'{file}_ECG.csv'), index = False)
                perc = (3.5 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))

                # Identify artifactual beats
                sqa = SQA.Cardio(fs)
                artifacts_ix = sqa.identify_artifacts(
                    beats_ix, method = artifact_method, tol = artifact_tol,
                    initial_hr = 'auto')
                ecg.loc[artifacts_ix, 'Artifact'] = 1

                # Compute IBIs and SQA metrics
                if ecg.columns[1] == 'Timestamp':
                    ibi = heartview.compute_ibis(
                        ecg, fs, beats_ix, 'Timestamp')
                    perc = (4 / total_progress) * 100
                    set_progress((perc * 100, f'{perc:.0f}%'))
                    metrics = sqa.compute_metrics(ecg,
                                                  beats_ix,
                                                  artifacts_ix,
                                                  ts_col = 'Timestamp',
                                                  seg_size = seg_size,
                                                  show_progress = False)
                else:
                    ibi = heartview.compute_ibis(ecg, fs, beats_ix)
                    perc = (4 / total_progress) * 100
                    set_progress((perc * 100, f'{perc:.0f}%'))
                    metrics = sqa.compute_metrics(ecg,
                                                  beats_ix,
                                                  artifacts_ix,
                                                  seg_size = seg_size,
                                                  show_progress = False)

                # Save to 'temp' directory
                ibi.to_csv(str(temp_path / f'{file}_IBI.csv'), index = False)
                metrics.to_csv(str(temp_path / f'{file}_SQA.csv'), index = False)

                # Downsample ECG data for quicker plot rendering
                ds_ecg, ds_ibi, ds_acc, ds_fs = utils._downsample_data(
                    ecg, fs, dtype, beats_ix, artifacts_ix, acc)
                fs = ds_fs
                perc = (5 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                ds_ecg.to_csv(str(render_dir / 'signal.csv'), index = False)
                ds_ibi.to_csv(str(render_dir / 'ibi.csv'), index = False)
                if ds_acc is not None:
                    ds_acc.to_csv(str(render_dir / 'acc.csv'), index = False)

            # Handle PPG CSV files
            if file_type == 'csv' and dtype == 'PPG':
                sqa = SQA.Cardio(fs)
                # If no timestamps are provided
                if d1 is None:
                    # If no acceleration data are provided
                    if (d3 is None) & (d4 is None) & (d5 is None):
                        ppg = utils._setup_data_samples(filepath, 'PPG', [d2])
                    else:
                        raw = utils._setup_data_samples(
                            filepath, 'PPG', [d2, d3, d4, d5])
                        ppg = raw[['Sample', 'ECG']].copy()
                        acc = raw[['Sample', 'X', 'Y', 'Z']].copy()
                else:
                    if (d3 is None) & (d4 is None) & (d5 is None):
                        ppg = utils._setup_data_ts(filepath, 'PPG', [d1, d2])
                    else:
                        raw = utils._setup_data_ts(
                            filepath, 'PPG', [d1, d2, d3, d4, d5])
                        ppg = raw[['Timestamp', 'PPG']].copy()
                        acc = raw[['Timestamp', 'X', 'Y', 'Z']].copy()

                # Pre-process acceleration data
                try:
                    acc['Magnitude'] = ACC.compute_magnitude(
                        acc['X'], acc['Y'], acc['Z'])
                    acc.to_csv(
                        str(temp_path / f'{file}_ACC.csv'), index = False)
                except:
                    acc = None

                # Filter PPG and detect beats
                detect_beats = PPG.BeatDetectors(fs)
                perc = (2 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                if filt_on:
                    filt = PPG.Filters(fs)
                    ppg['Filtered'] = filt.filter_signal(ppg['PPG'])
                    beats_ix = getattr(detect_beats, beat_detector)(ppg['Filtered'])
                else:
                    beats_ix = getattr(detect_beats, beat_detector)(ppg['PPG'])

                ppg.loc[beats_ix, 'Beat'] = 1
                ppg.insert(0, 'Segment', ppg.index // (seg_size * fs) + 1)
                ppg.to_csv(temp_path / f'{file}_PPG.csv', index = False)
                signal = ppg.copy()
                artifacts_ix = sqa.identify_artifacts(
                    beats_ix, method = artifact_method, tol = artifact_tol,
                    initial_hr = 'auto')
                signal.loc[artifacts_ix, 'Artifact'] = 1

                # Compute IBIs
                perc = (4 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                if signal.columns[1] == 'Timestamp':
                    ibi = heartview.compute_ibis(
                        ppg, fs, beats_ix, 'Timestamp')
                    perc = (5 / total_progress) * 100
                    set_progress((perc * 100, f'{perc:.0f}%'))
                    metrics = sqa.compute_metrics(ppg,
                                                  beats_ix,
                                                  artifacts_ix,
                                                  ts_col = 'Timestamp',
                                                  seg_size = seg_size,
                                                  show_progress = False)
                else:
                    ibi = heartview.compute_ibis(ppg, fs, beats_ix)
                    perc = (5 / total_progress) * 100
                    set_progress((perc * 100, f'{perc:.0f}%'))
                    metrics = sqa.compute_metrics(ppg,
                                                  beats_ix,
                                                  artifacts_ix,
                                                  seg_size = seg_size,
                                                  show_progress = False)

                ibi.to_csv(str(temp_path / f'{file}_IBI.csv'), index = False)
                metrics.to_csv(str(temp_path / f'{file}_SQA.csv'), index = False)

                # Downsample PPG data for quicker plot rendering
                ds_ppg, ds_ibi, ds_acc, ds_fs = utils._downsample_data(
                    ppg, fs, dtype, beats_ix, artifacts_ix, acc)
                fs = ds_fs
                ds_ppg.to_csv(str(render_dir / 'signal.csv'), index = False)
                ds_ibi.to_csv(str(render_dir / 'ibi.csv'), index = False)
                if ds_acc is not None:
                    ds_acc.to_csv(str(render_dir / 'acc.csv'), index = False)


            # Handle Empatica files
            if file_type == 'E4':
                perc = (2 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                E4 = heartview.Empatica(filepath)
                e4_data = E4.preprocess()

                # Accelerometer data
                acc = e4_data.acc
                acc.to_csv(str(render_dir / 'acc.csv'), index = False)
                acc.to_csv(str(temp_path / f'{file}_ACC.csv'), index = False)

                # EDA data
                eda = e4_data.eda
                eda.to_csv(str(temp_path / f'{file}_EDA.csv'), index = False)

                # BVP data
                bvp = e4_data.bvp
                bvp_fs = e4_data.bvp_fs
                sqa = SQA.Cardio(bvp_fs)

                # Extract PPG beats from Empatica E4 IBIs
                perc = (3 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                detect_beats = PPG.BeatDetectors(bvp_fs, False)
                e4_beats = detect_beats.adaptive_threshold(bvp['BVP'])
                ibi = heartview.compute_ibis(
                    bvp, bvp_fs, e4_beats, ts_col = 'Timestamp')
                ibi.to_csv(str(temp_path / f'{file}_IBI.csv'), index = False)
                bvp.loc[e4_beats, 'Beat'] = 1
                bvp.insert(0, 'Segment', bvp.index // (seg_size * fs) + 1)
                bvp.to_csv(str(temp_path / f'{file}_BVP.csv'), index = False)

                # Save render data
                signal = bvp.copy()
                artifacts_ix = sqa.identify_artifacts(
                    e4_beats, method = artifact_method, tol = artifact_tol,
                    initial_hr = 'auto')
                signal.loc[artifacts_ix, 'Artifact'] = 1
                signal.to_csv(str(render_dir / 'signal.csv'), index = False)
                ibi.to_csv(str(render_dir / 'ibi.csv'), index = False)

                # Compute SQA metrics
                perc = (5 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                metrics = sqa.compute_metrics(bvp,
                                              e4_beats,
                                              artifacts_ix,
                                              ts_col = 'Timestamp',
                                              seg_size = seg_size,
                                              show_progress = False)
                metrics.to_csv(
                    str(temp_path / f'{file}_SQA.csv'), index = False)

            # Store data variables in memory
            data['file type'] = file_type
            data['data type'] = dtype
            data['fs'] = fs
            data['filename'] = filename

            perc = (6 / total_progress) * 100
            set_progress((perc * 100, f'{perc:.0f}%'))

            return dtype_error, map_error, data

    # ===================== CONTROL DASHBOARD ELEMENTS ========================
    # === Toggle offcanvas ====================================================
    @app.callback(
        Output('offcanvas', 'is_open'),
        Input('reload-data', 'n_clicks')
    )
    def reload_data(n):
        """Open and close the offcanvas."""
        if n == 0:
            raise PreventUpdate
        else:
            return True

    # === Update SQA plots ====================================================
    @app.callback(
        Output('sqa-plot', 'figure'),
        [Input('memory-db', 'data'),
         Input('qa-charts-dropdown', 'value')]
    )
    def update_sqa_plot(data, sqa_view):
        """Update the SQA plot based on the selected view."""
        if data is None:
            raise PreventUpdate

        file = data['filename'].split('.')[0]
        sqa = pd.read_csv(str(temp_path / f'{file}_SQA.csv'))
        fs = int(data['fs'])
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
        return sqa_plot

    # === Update SQA table ====================================================
    @app.callback(
        [Output('device', 'children'),
         Output('filename', 'children'),
         Output('summary-table', 'children'),
         Output('segment-dropdown', 'options'),
         Output('export-summary', 'disabled')],
        Input('memory-db', 'data'),
    )
    def update_sqa_table(data):
        """Update the SQA summary table."""
        if data is None:
            raise PreventUpdate

        # Metadata
        file_type = data['file type']
        if file_type == 'E4':
            device = 'Empatica E4'
        elif file_type == 'Actiwave':
            device = 'Actiwave Cardio'
        else:
            device = 'Other'
        file = data['filename'].split('.')[0]
        filename = data['filename']
        if len(filename) > 27:
            filename = f'{filename[:27]} ...'
        data_type = data['data type']
        fs = int(data['fs'])

        sqa = pd.read_csv(str(temp_path / f'{file}_SQA.csv'))
        segments = sqa['Segment'].tolist()

        # Signal quality metrics
        if data_type in ['ECG', 'PPG', 'BVP']:
            cardio_sqa = SQA.Cardio(fs)
            table = cardio_sqa.display_summary_table(sqa)
        else:
            table = utils._blank_table()
        return device, filename, table, segments, False

    # === Update signal plots =================================================
    @app.callback(
        [Output('raw-data', 'figure'),
         Output('segment-dropdown', 'value'),
         Output('prev-n-tooltip', 'is_open'),
         Output('next-n-tooltip', 'is_open'),
         Output('beat-correction-status', 'data')],
        Input('memory-db', 'data'),
        Input('segment-dropdown', 'value'),
        Input('prev-segment', 'n_clicks'),
        Input('next-segment', 'n_clicks'),
        Input('beat-correction', 'n_clicks'),
        State('beat-correction-status', 'data'),
        State('seg-size', 'value'),
        State('segment-dropdown', 'options')
    )
    def update_signal_plots(data, selected_segment, prev_n, next_n, beat_correction_n,
                            beat_correction_status, segment_size, segments):
        """Update the raw data plot based on the selected segment view."""
        if data is None:
            raise PreventUpdate
        else:
            file = data['filename'].split('.')[0]
            data_type = data['data type']
            signal = pd.read_csv(str(render_dir / 'signal.csv'))
            fs = int(data['fs'])
            seg_size = int(segment_size)

            # Reset selected_segment to 1 when new data is loaded
            if ctx.triggered_id == 'memory-db' or selected_segment is None:
                selected_segment = 1

            # If cardiac data was run
            if data_type in ['ECG', 'PPG', 'BVP']:
                ibi = pd.read_csv(str(render_dir / 'ibi.csv'))
                try:
                    acc = pd.read_csv(str(render_dir / 'acc.csv'))
                except FileNotFoundError:
                    acc = None

                y_axis = data_type
                x_axis = 'Timestamp' if 'Timestamp' in signal.columns else \
                    'Sample'
                prev_tt_open = False
                next_tt_open = False

                if ctx.triggered_id == 'prev-segment':
                    if selected_segment != 1:
                        prev_tt_open = False
                        next_tt_open = False
                        selected_segment -= 1
                    else:
                        prev_tt_open = True
                elif ctx.triggered_id == 'next-segment':
                    if selected_segment != max(segments):
                        prev_tt_open = False
                        next_tt_open = False
                        selected_segment += 1
                    else:
                        next_tt_open = True
                # Correct beats if beat correction button is clicked
                # TODO: Update ibi and ibi_corrected accordingly
                if ctx.triggered_id == 'beat-correction':
                    beats_ix = signal.loc[signal['Beat'] == 1].index.tolist()
                    sqa = SQA.Cardio(fs)
                    beats_ix_corrected, corrected_ibis, original, corrected = sqa.correct_interval(beats_ix, seg_size=seg_size, print_estimated_hr=False)
                    signal.loc[beats_ix_corrected, 'Corrected'] = 1
                    signal.to_csv(str(render_dir / 'signal.csv'), index = False)
                    ibi_corrected = heartview.compute_ibis(
                        signal, fs, beats_ix_corrected, 'Timestamp')
                    ibi_corrected.to_csv(str(temp_path / f'{file}_IBI_corrected.csv'), index = False)
                    beat_correction_status['status'] = 'suggested'
                else:
                    if beat_correction_status['status'] == 'suggested':
                        ibi_corrected = pd.read_csv(str(temp_path / f'{file}_IBI_corrected.csv'))
                    else:
                        ibi_corrected = None

                # Create the signal subplots
                overlay_corrected = beat_correction_status['status'] == 'suggested'
                signal_plots = heartview.plot_cardio_signals(
                    signal, fs, ibi, data_type,
                    x_axis, y_axis, ibi_corrected, acc, selected_segment, seg_size, overlay_corrected=overlay_corrected)

            # If EDA data was run
            else:
                signal_plots = utils._blank_fig()

            return signal_plots, selected_segment, prev_tt_open, next_tt_open, beat_correction_status

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
            State('export-type', 'value'),
            State('memory-db', 'data'),
        ],
        background = True,
        running = [
            (Output('export-progress-bar', 'style'),
             {'visibility': 'visible'}, {'visibility': 'hidden'}),
            (Output('ok-export', 'disabled'), True, False),
        ],
        cancel = [
            Input('close-export', 'n_clicks')],
        progress = [
            Output('export-progress-bar', 'value'),
            Output('export-progress-bar', 'label')
        ],
        prevent_initial_call = True
    )
    def export_summary(set_progress, n, done, export_type, data):
        """Export the SQA summary file and confirm the export."""
        if ctx.triggered_id in ('close-export', 'close-export2'):
            set_progress((0, ''))
            return [False, None, True, False, True]
        else:
            file = data['filename'].split('.')[0]
            data_type = data['data type']
            files = [str(temp_path / f'{file}_SQA.csv')]
            if data_type == 'BVP':  # if data is from the Empatica E4
                files.extend([
                    temp_path / f'{file}_BVP.csv',
                    temp_path / f'{file}_IBI.csv',
                    temp_path / f'{file}_EDA.csv',
                ])
            elif data_type == 'Actiwave':
                files.extend([temp_path / f'{file}_ECG.csv',
                              temp_path / f'{file}_IBI.csv'])
            elif data_type == 'PPG':
                files.extend([temp_path / f'{file}_PPG.csv',
                              temp_path / f'{file}_IBI.csv'])
            else:  # if data_type == 'csv'
                files.extend([temp_path / f'{file}_ECG.csv',
                              temp_path / f'{file}_IBI.csv'])
            if (temp_path / f'{file}_ACC.csv').exists():
                files.append(temp_path / f'{file}_ACC.csv')

            # Set up progress bar for export process
            increments = len(files)
            step = 100 / increments
            next_progress_threshold = step  # start at the first threshold

            export = None
            if export_type == 'Zip':
                output_zip = BytesIO()
                with zipfile.ZipFile(output_zip, 'w') as archive:
                    for i, csv in enumerate(files):
                        file_name = path.basename(csv)
                        with open(csv, 'rb') as f:
                            archive.writestr(file_name, f.read())
                        current_progress = (i + 1) / len(files) * 100
                        if (current_progress < 100 or i <= len(files)):
                            perc = next_progress_threshold
                            set_progress((perc, f'{perc:.0f}%'))
                            next_progress_threshold += step
                            sleep(0.5)
                output_zip.seek(0)
                export = send_bytes(output_zip.getvalue(),
                                    f'{file}_sqa_summary.zip')
            elif export_type == 'Excel':
                output_xls = BytesIO()
                max_rows = 1_000_000  # row limit
                with pd.ExcelWriter(output_xls) as xlsx:
                    for i, csv in enumerate(files):
                        df = pd.read_csv(csv)
                        fname = Path(csv).stem.split('_')[-1]
                        num_sheets = (len(df) + max_rows - 1) // max_rows

                        for j in range(num_sheets):
                            start_row = j * max_rows
                            end_row = min((j + 1) * max_rows, len(df))
                            df_chunk = df.iloc[start_row:end_row]

                            # prevent Excel from writing past max row
                            if df_chunk.empty:
                                continue

                            sheet_name = f'{fname}_{j + 1}' if num_sheets > 1 else fname
                            sheet_name = sheet_name[:31]  # sheet name limit
                            df_chunk.to_excel(xlsx, sheet_name = sheet_name,
                                              index = False)

                        current_progress = (i + 1) / len(files) * 100
                        if (current_progress < 100 or i <= len(files)):
                            perc = next_progress_threshold
                            set_progress((perc, f'{perc:.0f}%'))
                            next_progress_threshold += step
                output_xls.seek(0)
                export = send_bytes(output_xls.getvalue(),
                                    f'{file}_sqa_summary.xlsx')

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