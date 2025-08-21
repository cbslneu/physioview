from dash import html, dcc
from heartview.dashboard import utils
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_uploader as du
import uuid

layout = html.Div(id = 'main', children = [

    # OFFCANVAS
    dbc.Offcanvas(children = [
        dcc.Store(id = 'memory-load', storage_type = 'memory'),
        dcc.Store(id = 'config-memory', storage_type = 'memory'),
        dcc.Store(id = 'config-download-memory', storage_type = 'memory'),
        dcc.Store(id = 'memory-db', storage_type = 'memory'),
        html.Span(className = 'h5',
                  children = ['Welcome to HeartView']),
        html.P(children = [
            'Explore signal quality metrics for ambulatory cardiac '
            'data collected with devices such as the Empatica E4 or Actiwave '
            'Cardio.']),
        html.H4(children = [
            'Load Data',
            html.I(className = 'fa-regular fa-circle-question',
                   style = {'marginLeft': '5px'},
                   id = 'load-data-help')],
                style = {'display': 'flex', 'alignItems': 'center'}),
        dbc.Tooltip(
            'Valid file types: .edf (Actiwave); .zip (E4); .csv (other '
            'ECG/PPG sources).',
            target = 'load-data-help'),
        du.Upload(id = 'dash-uploader',
                  text = 'Select Data File...',
                  filetypes = ['EDF', 'edf', 'zip', 'csv'],
                  upload_id = uuid.uuid4(),
                  max_file_size = 1024,
                  text_completed = '',
                  cancel_button = True,
                  default_style = {'lineHeight': '50px',
                                   'minHeight': '50px',
                                   'borderRadius': '5px'}),
        html.Div(id = 'file-check', children = []),
        daq.BooleanSwitch(
            id = 'toggle-config',
            color = '#ee8a78',
            label = 'Load a Configuration File',
            labelPosition = 'right',
            on = False
        ),
        html.Div(id = 'config-upload-div', hidden = True, children = [
            du.Upload(id = 'config-uploader',
                      text = 'Select Configuration File...',
                      filetypes = ['json', 'JSON'],
                      upload_id = 'cfg',
                      text_completed = '',
                      cancel_button = True)]),
        html.Div(id = 'setup-data', hidden = True, children = [
            html.H4(children = [
                'Setup Data',
                html.I(className = 'fa-regular fa-circle-question',
                       style = {'marginLeft': '5px'},
                       id = 'data-var-help')],
                style = {'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip('Set your sampling rate and/or map the headers in '
                        'your file to the corresponding data variables.',
                        target = 'data-var-help'),
            html.Div(id = 'data-type-container', children = [
                html.Span('Data Type: '),
                dbc.RadioItems(
                    id = 'data-types',
                    options = [
                        {'label': 'ECG', 'value': 'ECG'},
                        {'label': 'PPG', 'value': 'PPG'},
                    ],
                    inline = True
                )]
            ),
            html.Span('Sampling Rate: '),
            dcc.Input(id = 'sampling-rate', value = 500, type = 'number',
                      style = {'display': 'inline', 'marginLeft': '3px'}),
            # ========================== TO DO ===============================
            html.Div(id = 'resample', hidden = True, children = [
                daq.BooleanSwitch(
                    id = 'toggle-resample',
                    color = '#ee8a78',
                    label = 'Resample:',
                    labelPosition = 'right',
                    on = False
                ),
                dcc.Input(id = 'resampling-rate', value = 64,
                          type = 'number', disabled = True)
            ]),
            # ================================================================
            html.Div(id = 'data-variables', children = [
                html.Div(id = 'variables-header', children = [
                    html.Span('Variables:', style = {'display': 'inlineBlock'}),
                    html.Span(hidden = True, children = [
                        html.I(className = 'fa-solid fa-circle-exclamation',
                               style = {'fontSize': '12px',
                                        'marginRight': '3px'}),
                        'Header mismatch in files!'],
                        id = 'variable-mapping-check')
                ]),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(id = 'data-type-dropdown-1',
                                     placeholder = 'Time/Sample',
                                     options = ['<Var>', '<Var>']),
                        id = 'setup-timestamp'),
                    dbc.Col(
                        dcc.Dropdown(id = 'data-type-dropdown-2',
                                     placeholder = 'Signal',
                                     options = ['<Var>', '<Var>']),
                        id = 'setup-cardio'),
                    dbc.Col(
                        dcc.Dropdown(id = 'data-type-dropdown-3',
                                     placeholder = 'X',
                                     options = ['<Var>', '<Var>'])),
                    dbc.Col(
                        dcc.Dropdown(id = 'data-type-dropdown-4',
                                     placeholder = 'Y',
                                     options = ['<Var>', '<Var>'])),
                    dbc.Col(
                        dcc.Dropdown(id = 'data-type-dropdown-5',
                                     placeholder = 'Z',
                                     options = ['<Var>', '<Var>'])),
                ], className = 'setup-data-dropdowns')
            ]),
        ]),
        html.Div(id = 'preprocess-data', hidden = True, children = [
            html.H4('Preprocess Data'),
            html.Div(id = 'filter-data', children = [
                daq.BooleanSwitch(
                    id = 'toggle-filter',
                    color = '#ee8a78',
                    label = 'Filter Signal',
                    labelPosition = 'left',
                    on = False),
                html.I(className = 'fa-regular fa-circle-question',
                       id = 'filter-help'),
                dbc.Tooltip('Apply a filter to remove low- and high-frequency '
                            'noise, including baseline drift, powerline '
                            'interference, and muscle activity.',
                            target = 'filter-help',
                            style = {'--bs-tooltip-max-width': '225px'})
            ]),
            html.Div(id = 'cardio-preprocessing', hidden = True, children = [
                html.Div(id = 'artifact-params', children = [
                    html.Div(children = [
                        html.Span('Artifact Identification Method:'),
                        html.Button(
                            html.I(className = 'fa-regular fa-circle-question'),
                            id = 'artifact-method-help', n_clicks = 0),
                        dcc.Dropdown(
                            id = 'artifact-method',
                            options = [
                                {'label': 'Berntson et al. (1990)',
                                 'value': 'cbd'},
                                {'label': 'Hegarty-Craver et al. (2018)',
                                 'value': 'hegarty'}
                            ],
                            value = 'cbd', clearable = False)
                    ]),
                    html.Div(children = [
                        html.Span('Artifact Identification Tolerance:'),
                        dcc.Input(id = 'artifact-tol', value = 1, type = 'number',
                                  min = 0.1, max = 2, step = 0.1),
                        html.I(className = 'fa-regular fa-circle-question',
                               id = 'artifact-tol-help'),
                        dbc.Tooltip(
                            'This sets the tolerance level of the artifact '
                            'detection algorithm. A lower tolerance will lead '
                            'to more artifacts flagged.', target = 'artifact-tol-help')
                    ], style = {'display': 'flex', 'alignItems': 'center'})
                ]),
                html.Div(id = 'select-beat-detector', children = [
                    html.Span('Beat Detector:'),
                    dcc.Dropdown(id = 'beat-detectors',
                                 options = [], clearable = False)
                ]),
            ]),
            html.Div(id = 'segment-data', children = [
                html.Span('Segment Size (sec):'),
                dcc.Input(id = 'seg-size', value = 60, type = 'number'),
                html.I(className = 'fa-regular fa-circle-question',
                       id = 'seg-data-help'),
                dbc.Tooltip(
                    'This sets the size of the windows into which your data '
                    'will be segmented; by default, 60 seconds.',
                    target = 'seg-data-help')
            ]),
        ]),
        html.Div(id = 'run-save-buttons', children = [
            html.Button('Run', n_clicks = 0, id = 'run-data', disabled = True),
            html.Button('Stop', n_clicks = 0, id = 'stop-run', hidden = True,
                        disabled = False),
            html.Button('Save', n_clicks = 0, id = 'configure',
                        disabled = True)
        ]),

        # Variable Mappings Validator
        dbc.Modal([
            dbc.ModalBody([
                html.Div(className = 'validation-content', children = [
                    html.I(className = 'fa-solid fa-circle-xmark'),
                    html.Div(className = 'validation-error', children = [
                        html.H5('Missing variable mappings'),
                        html.P('Please ensure at least both \'Time/Sample\' '
                               'and \'Signal\' are mapped to your data.')]),
                    html.Button(
                        html.I(className = 'fa-solid fa-xmark'),
                        id = 'close-mapping-validator')
                ])
            ])
        ], className = 'validation-error-modal',
            id = 'mapping-validator', is_open = False, centered = True),

        # Data Type Input Validator
        dbc.Modal([
            dbc.ModalBody([
                html.Div(className = 'validation-content', children = [
                    html.I(className = 'fa-solid fa-circle-xmark'),
                    html.Div(className = 'validation-error', children = [
                        html.H5('Missing data type'),
                        html.P('Please ensure a data type is selected.')]),
                    html.Button(
                        html.I(className = 'fa-solid fa-xmark'),
                        id = 'close-dtype-validator')
                ])
            ])
        ], className = 'validation-error-modal',
            id = 'dtype-validator', is_open = False, centered = True),

        # Artifact Identification Method Information
        dbc.Modal([
            dbc.ModalBody([
                html.Div(className = 'information-content', children = [
                    html.I(className = 'fa-regular fa-circle-question',
                           style = {'color': '#333333'}),
                    html.Div(className = 'artifact-method-info', children = [
                        html.H5('Artifact Identification Methods'),
                        html.P(children = [
                            html.Span('Berntson et al. (1990)',
                                      style = {'fontWeight': 600}),
                            html.Span('This criterion beat difference '
                                      'approach detects artifacts by '
                                      'flagging heartbeats that deviate '
                                      'significantly from nearby intervals, '
                                      'using a threshold based on the '
                                      'variability of surrounding beats.')]),
                        html.P(children = [
                            html.Span('Hegarty-Craver et al. (2018)',
                                      style = {'fontWeight': 600}),
                            html.Span('This approach detects artifacts by '
                                      'comparing each interbeat interval to '
                                      'an estimate based on nearby intervals, '
                                      'flagging beats as artifacts if they '
                                      'fall outside a predefined acceptable '
                                      'range.')])
                    ]),
                    html.Button(
                        html.I(className = 'fa-solid fa-xmark'),
                        id = 'close-artifact-method-info')
                ])
            ])
        ], className = 'validation-error-modal',
            id = 'artifact-identification-modal', is_open = False,
            centered = True),

        # Configuration File Exporter
        dbc.Modal(id = 'config-modal', is_open = False, children = [
            dbc.ModalHeader(dbc.ModalTitle('Export Configuration')),
            dbc.ModalBody(children = [
                html.Div(id = 'config-description', children = [
                    html.Div('Save your parameters in a JSON configuration '
                             'file.'),
                    html.Div(children = [
                        html.Span('Filename: '),
                        dcc.Input(id = 'config-filename', value = 'config1',
                                  type = 'text'),
                        html.Span(' .json')], style = {'marginTop': '20px'})
                ]),
                html.Div(id = 'config-check', children = [
                    html.I(className = 'fa-solid fa-circle-check',
                       style = {'color': '#63e6be', 'fontSize': '26pt'}),
                    html.P('Configuration file created.',
                           style = {'marginTop': '5px'})
                ], hidden = True)
            ]),
            dbc.ModalFooter(children = [
                html.Div(id = 'config-modal-btns', children = [
                    html.Button('Configure', n_clicks = 0, id = 'config-btn'),
                    dcc.Download(id = 'config-file-download'),
                    dbc.Button('Cancel', n_clicks = 0, id = 'close-config1')],
                         hidden = False),
                html.Div(id = 'config-close-btn', children = [
                    dbc.Button('Done', n_clicks = 0, id = 'close-config2')],
                         hidden = True)
            ], style = {'display': 'inline'})
        ], backdrop = 'static', centered = True),

        # SQA Pipeline Progress Indicator
        dbc.Progress(id = 'progress-bar', animated = True)
    ], id = 'offcanvas', style = {'width': 450}, is_open = True),

    # NAVIGATION BAR
    html.Div(className = 'banner', children = [
       dbc.Row(children = [
           dbc.Col(html.Img(src = './assets/heartview-logo.png',
                            className = 'logo')),
           dbc.Col(html.Div(html.A(
               'Documentation',
               href = 'https://heartview.readthedocs.io/en/latest/',
               target = '_blank')),
                   width = 3),
           dbc.Col(html.Div(html.A(
               'Contribute',
               href = 'https://heartview.readthedocs.io/en/latest/contribute.html',
               target = '_blank')),
                   width = 3)
       ], justify = 'between')
    ]),

    # MAIN DASHBOARD
    html.Div(className = 'app', children = [

        html.Div(id = 'qa-section', children = [

            # Subject Selection Dropdown
            html.Div(id = 'select-subject', children = [
                html.I(className = 'fa-solid fa-user'),
                dcc.Dropdown(
                    id = 'subject-dropdown',
                    placeholder = 'Select Data',
                    options = [],
                    value = None,
                    disabled = True,
                    clearable = False
                ),
            ]),

            # Panel 1: Data Summary
            html.Div(className = 'data-summary', children = [
                html.Div(id = 'data-summary-header', children = [
                    html.H2('Quality Summary'),
                    html.I(className = 'fa-solid fa-pen-to-square',
                           id = 'reload-data'),
                    dbc.Tooltip('Reload data file', target = 'reload-data')],
                    style = {'display': 'flex',
                             'justifyContent': 'spaceBetween'}),
                html.Div(className = 'data-summary-divs', id = 'summary-table',
                         children = [utils._blank_table()]),

                # Data Summary Exporter
                html.Div(children = [
                    html.Button('Export Summary',
                                id = 'export-summary',
                                n_clicks = 0,
                                disabled = True),
                    dbc.Modal(
                        id = 'export-modal',
                        is_open = False,
                        children = [
                            dbc.ModalHeader(
                                dbc.ModalTitle(children = [
                                    'Export Summary',
                                ])
                            ),
                                dbc.ModalBody(children = [
                                    html.Div(
                                        id = 'export-description',
                                        children = [
                                            html.Div([
                                                html.Span('''Download summary data as a Zip archive file (.zip) 
                                                or Excel (.xlsx) file for a single file or batch of files.'''),
                                                html.Span(''' Excel files may take longer to write.''',
                                                          style = {'fontStyle': 'italic', 'color': '#de7765'})]
                                            ),
                                            html.Div(className = 'export-options', children = [
                                                html.Div([
                                                    html.Label('Export Mode', htmlFor = 'export-mode'),
                                                    html.I(className = 'fa-regular fa-circle-question',
                                                           style = {'marginLeft': '5px'},
                                                           id = 'export-mode-help'),
                                                    dbc.Tooltip(
                                                        "'Single' exports summary data of the current file. "
                                                        "'Batch' exports data for all preprocessed subjects.",
                                                        target = 'export-mode-help'),
                                                    dbc.RadioItems(
                                                        id = 'export-mode',
                                                        options = [
                                                            {'label': 'Single', 'value': 'Single'},
                                                            {'label': 'Batch', 'value': 'Batch', 'disabled': True}
                                                        ],
                                                        inline = True)
                                                ], style = {'display': 'inline-block'}),
                                                html.Div(children = [
                                                    html.Label('File Format', htmlFor = 'export-type'),
                                                    html.I(className = 'fa-regular fa-circle-question',
                                                                       style = {'marginLeft': '5px'},
                                                                       id = 'export-format-help'),
                                                    dbc.Tooltip(
                                                        "'Zip' creates a .zip file containing CSVs and a summary "
                                                        "text file. 'Excel' generates a workbook. In 'Batch' mode, all "
                                                        "subjects' Excel files are bundled into a single .zip archive.",
                                                        target = 'export-format-help',
                                                    style = {'--bs-tooltip-max-width': '225px'}),
                                                    dbc.RadioItems(
                                                        ['Zip', 'Excel'],
                                                        inline = True,
                                                        id = 'export-type')
                                                ], style = {'display': 'inline-block'})])
                                            ]),
                                    dbc.Progress(id = 'export-progress-bar',
                                                 animated = True),
                                    html.Div(id = 'export-confirm',
                                             children = [
                                                 html.I(className = 'fa-solid fa-circle-check',
                                                        style = {'color': '#63e6be', 'fontSize': '26pt'}),
                                                 html.P('Summary file created.',
                                                        style = {'marginTop': '5px'})],
                                             hidden = True)]),
                                # ]),
                            dbc.ModalFooter(children = [
                                html.Div(id = 'export-modal-btns',
                                         children = [
                                             html.Button('OK', n_clicks = 0,
                                                         id = 'ok-export',
                                                         disabled = True),
                                             dcc.Download(id = 'download-summary'),
                                             dbc.Button('Cancel', n_clicks = 0, id = 'close-export'),]),
                                html.Div(id = 'export-close-btn',
                                         children = [
                                             dbc.Button('Done', n_clicks = 0, id = 'close-export2')],
                                         hidden = True)],
                                style = {'display': 'inline'})
                        ], backdrop = 'static', centered = True)])
            ]),

            # Panel 2: Data Quality
            html.Div(className = 'qa', children = [
                html.Div(className = 'qa-charts', children = [
                    html.H2('Data Quality'),
                    html.Div(className = 'qa-charts-selection', children = [
                        html.I(className = 'fa-solid fa-eye'),
                        dcc.Dropdown(
                            id = 'qa-charts-dropdown',
                            options = [
                                {'label': 'Missing Beats', 'value': 'missing'},
                                {'label': 'Artifact Beats',
                                 'value': 'artifact'}],
                            value = 'missing',  # default chart
                            clearable = False
                        ),
                    ]),
                ]),
                html.Div(className = 'figs', children = [
                    dcc.Loading(
                        type = 'circle', color = '#3a4952', children = [
                            dcc.Graph(
                                id = 'sqa-plot',
                                figure = utils._blank_fig('pending'))])
                ])
            ])
        ]),

        # Bottom Panel: Raw ECG/BVP, IBI, and ACC Plots
        html.Div(className = 'raw-plots', children = [
            dcc.Store(id = 'beat-correction-status', storage_type = 'memory', data = {'status': None}),
            html.Div(className = 'graph-settings', children = [
                html.Div(id = 'graph-settings-left', children = [
                    html.H4('Signal View'),
                    html.Div(className = 'segment-view', children = [
                        html.Span('|', className = 'separator'),
                        html.H5('Segment:'),
                        html.Button(
                            html.I(className = 'fa-solid fa-chevron-left'),
                            id = 'prev-segment'),
                        dbc.Tooltip('No more segments.', id = 'prev-n-tooltip',
                                    target = 'prev-segment', is_open = False,
                                    trigger = None, placement = 'top-start'),
                        dcc.Dropdown(id = 'segment-dropdown',
                                     placeholder = '1',
                                     value = 1,
                                     options = [{'label': '1', 'value': 1}],
                                     clearable = False),
                        html.Button(
                            html.I(className = 'fa-solid fa-chevron-right'),
                            id = 'next-segment'),
                        dbc.Tooltip('No more segments.', id = 'next-n-tooltip',
                                    target = 'next-segment', is_open = False,
                                    trigger = None, placement = 'top-start'),
                    ])
                ]),
                html.Div(className = 'processing-buttons', children = [
                    html.Button(children = [
                        html.I(className = 'fa-solid fa-wand-magic-sparkles'),
                        html.Span('Correct Beats')
                    ], id = 'beat-correction', hidden = False),
                    html.Button(children = [
                        html.I(className = 'fa-solid fa-circle-check'),
                        html.Span('Accept Corrections')
                    ], id = 'accept-corrections', hidden = True),
                    html.Button(children = [
                        html.I(className = 'fa-solid fa-circle-xmark'),
                        html.Span('Reject Corrections')
                    ], id = 'reject-corrections', hidden = True),
                    html.Button(children = [
                        html.I(className = 'fa-solid fa-rotate-left'),
                        html.Span('Revert Corrections')
                    ], id = 'revert-corrections', hidden = True),
                    html.Div(id = 'beat-editor-option', children = [
                        html.Span('|', className = 'separator'),
                        html.Button(children = [
                        dbc.Spinner(
                            children = [
                                html.I(className = 'fa-solid fa-arrow-up-right-from-square')
                            ], id = 'beat-editor-spinner', size = 'sm'),
                            html.Span('Beat Editor', id = 'beat-editor-btn-label'),
                        ], id = 'open-beat-editor', n_clicks = 0,
                            disabled = True),
                    ]),
                    html.Div(id = 'postprocess-option', children = [
                        html.Span('|', className = 'separator'),
                        html.Button(children = [
                            html.I(className = 'fa-solid fa-hammer'),
                            html.Span('Postprocess')
                        ], id = 'postprocess-data', n_clicks = 0,
                            disabled = True),
                    ])
                ])
            ]),
            html.Div(className = 'figs', children = [
                dcc.Loading(type = 'circle', color = '#3a4952', children = [
                    dcc.Graph(id = 'raw-data',
                              figure = utils._blank_fig('pending'),
                              style = {'height': 'auto'})
                ])
            ])
        ]),

        # Beat Editor Modal
        dcc.Store(id = 'beat-editor-iframe-listener'),
        html.Div(id = 'be-create-file', style = {'display': 'none'}),
        html.Div(id = 'be-edited-trigger', style = {'display': 'none'},
                 children = [None]),
        dbc.Modal(id = 'beat-editor-modal', children = [
            dbc.ModalHeader(dbc.ModalTitle('Beat Editor'), close_button = False),
            dbc.ModalBody(children = [
                dcc.Loading(type = 'circle', color = '#3a4952', children = [
                    html.Div(id = 'beat-editor-content', children = [])
                ])
            ]),
            dbc.ModalFooter(children = [
                html.Div(id = 'beat-editor-buttons', children = [
                    html.Button('Cancel', id = 'cancel-beat-edits'),
                    html.Button('Apply Edits', id = 'ok-beat-edits'),
                    dbc.Tooltip(children = [
                        html.I(className = 'fa-solid fa-triangle-exclamation',
                               style = {'marginRight': '5px'}),
                        "Be sure to click 'Save' in the Beat Editor before continuing."],
                        target = 'ok-beat-edits', placement = 'top', is_open = False,
                        style = {'--bs-tooltip-max-width': '225px'},
                        autohide = False),
                ]),
            ]),
        ], is_open = False, centered = True, backdrop = True),

        # Postprocess Modal
        dbc.Modal(id = 'postprocess-modal', children = [
            dbc.ModalHeader(dbc.ModalTitle('Postprocess Data')),
            dbc.ModalBody(children = [
                html.Span('''Postprocess data from a single file or a batch 
                of files.'''),
                html.Div(id = 'postprocess-outputs', children = [
                    html.Label('Output Type', htmlFor = 'postprocess-options'),
                    html.Span(style = {'fontSize': '9pt', 'marginLeft': '5px'},
                              children = [
                                  '[', html.A('Select All', id = 'select-all'), ']']),
                    dbc.Checklist(id = 'postprocess-options', options = [
                        {'label': 'Raw and Cleaned Data', 'value': 'signal_data'},
                        {'label': 'Interval Series', 'value': 'interval_data'},
                        {'label': 'Derived Features', 'value': 'features'},
                        {'label': 'Signal Quality Metrics', 'value': 'sqa'}
                    ], value = [], inline = True),
                ]),
                html.Hr(className = 'light-divider'),
                html.Div(id = 'postprocess-params-container', children = [
                    html.Label('Output Parameters', style = {'fontWeight': 600}),
                    html.I(className = 'fa-regular fa-circle-question',
                           style = {'marginLeft': '5px'},
                           id = 'output-params-help'),
                    dbc.Tooltip('''Set the window length and step size for 
                    segmenting your signal when extracting derived features.''',
                        target = 'output-params-help'),
                    html.Div(id = 'postprocess-params', children = [
                        html.Div(id = 'postprocess-params-left', children = [
                            html.Span('Window Size (sec):'),
                            dcc.Input(id = 'feature-window-size', value = 60,
                                      type = 'number')
                        ]),
                        html.Div(id = 'postprocess-params-right', children = [
                            html.Span('Step Size (sec):'),
                            dcc.Input(id = 'feature-step-size', value = 1,
                                      type = 'number')
                        ]),
                    ]),
                ]),
                html.Hr(className = 'light-divider'),
                html.Div(className = 'export-options', children = [
                    html.Div([
                        html.Label('Export Mode', htmlFor = 'export-mode'),
                        html.I(className = 'fa-regular fa-circle-question',
                               style = {'marginLeft': '5px'},
                               id = 'postprocess-export-mode-help'),
                        dbc.Tooltip(
                            "'Single' exports summary data of the current file. "
                            "'Batch' exports data for all preprocessed subjects.",
                            target = 'postprocess-export-mode-help'),
                        dbc.RadioItems(
                            id = 'postprocess-export-mode',
                            options = [
                                {'label': 'Single', 'value': 'single'},
                                {'label': 'Batch', 'value': 'batch', 'disabled': True}
                            ],
                            inline = True)
                    ]),
                    html.Div(children = [
                        html.Label('File Format', htmlFor = 'export-type'),
                        html.I(className = 'fa-regular fa-circle-question',
                                           style = {'marginLeft': '5px'},
                                           id = 'postprocess-export-format-help'),
                        dbc.Tooltip(
                            "'Zip' creates a .zip file containing CSVs and a summary "
                            "text file. 'Excel' generates a workbook. In 'Batch' mode, all "
                            "subjects' Excel files are bundled into a single .zip archive.",
                            target = 'postprocess-export-format-help',
                        style = {'--bs-tooltip-max-width': '225px'}),
                        dbc.RadioItems(
                            options = [
                                {'label': 'Zip', 'value': 'zip'},
                                {'label': 'Excel', 'value': 'excel'}],
                            inline = True,
                            id = 'postprocess-export-type')
                    ])
                ])
            ]),
            dbc.ModalFooter(children = [
                dbc.Progress(id = 'postprocess-progress-bar',
                             animated = True),
                html.Div(id = 'postprocess-modal-buttons', children = [
                    html.Button('Cancel', id = 'cancel-postprocess'),
                    html.Button('Postprocess', id = 'ok-postprocess'),
                    dcc.Download(id = 'download-postprocess'),
                ]),
            ])
        ], is_open = False, centered = True, backdrop = True),
        dbc.Toast(id = 'postprocess-done-toast',
                  children = ['Data postprocessed and exported.'],
                  header = html.Div([
                      html.I(className = 'fa-solid fa-circle-check',
                             style = {'color': '#3ab540', 'fontSize': '14pt'}),
                      html.Span('Complete')],
                      style = {'display': 'flex', 'gap': '0.5em',
                               'alignItems': 'center'}),
                  dismissable = True, is_open = False,
                  style = {
                      'position': 'fixed',
                      'top': '45%',
                      'left': '50%',
                      'transform': 'translate(-50%, -45%)',
                      'width': 350,
                      'zIndex': 2000
                  }),
        dbc.Toast(id = 'postprocess-error-toast',
                  children = ['Missing postprocessing parameters.'],
                  header = html.Div([
                      html.I(className = 'fa-solid fa-circle-xmark',
                             style = {'fontSize': '14pt'}),
                      html.Span('Error')],
                      style = {'display': 'flex', 'gap': '0.5em',
                               'alignItems': 'center'}),
                  dismissable = True, duration = 3000, is_open = False,
                  style = {
                      'position': 'fixed',
                      'top': '45%',
                      'left': '50%',
                      'transform': 'translate(-50%, -45%)',
                      'width': 350,
                      'zIndex': 2000
                  })
    ]),

    # FOOTER
    html.Div(className = 'bottom', children = [
        html.Span('Created by the '),
        html.A(href = 'https://github.com/cbslneu', children = [
            'Computational Behavioral Science Lab'], target = 'new'),
        html.Span(', Northeastern University')
    ])
])