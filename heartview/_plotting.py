import numpy as np
import plotly.graph_objects as go

__all__ = ['_acc_subplot', '_ibi_subplot',
           '_DEFAULT_SIGNAL_PARAMS', '_EDIT_STYLES']

_DEFAULT_SIGNAL_PARAMS = {
    'ECG':  {'unit': 'mV', 'color': '#3562bd', 'peak': 'Detected Beat'},
    'PPG':  {'unit': 'bvp', 'color': '#3562bd', 'peak': 'Detected Beat'},
    'BVP':  {'unit': 'bvp', 'color': '#3562bd', 'peak': 'Detected Beat'},
    'EDA':  {'unit': 'µS', 'color': '#249ab5', 'peak': 'SCR'},
    'HR':   {'unit': 'bpm', 'color': '#3562bd', 'peak': 'Max HR'},
    'RESP': {'unit': 'a.u.', 'color': '#d16296', 'peak': 'Peak'},
    'TEMP': {'unit': '°C', 'color': '#8659c2', 'peak': 'Peak'},
}

_EDIT_STYLES = {
    'Add':      {'mode': 'markers',
                 'marker': {'color': '#02e337', 'size': 8},
                 'name': 'Added Beat'},
    'Delete':   {'mode': 'markers',
                 'marker': {'symbol': 'x', 'color': '#c70831', 'size': 8},
                 'name': 'Deleted Beat'},
    'Unusable': {'mode': 'lines',
                 'line': {'color': '#969696', 'width': 1.75},
                 'name': 'Unusable Data'}
}

def _acc_subplot(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    fig: go.Figure,
    line_dict: dict = dict(color = 'forestgreen', width = 1.5),
    name: str = 'ACC'
) -> go.Figure:
    fig.add_trace(
        go.Scatter(
            x = acc_x,
            y = acc_y,
            name = name,
            line = line_dict,
            hovertemplate = f'<b>{name}</b>: %{y:.2f} m/s² <extra></extra>'),
        row = 1, col = 1)
    fig.update_yaxes(
        title_text = 'm/s²', title_standoff = 5,
        showgrid = True, gridwidth = 0.5, gridcolor = 'lightgrey',
        griddash = 'dot', tickcolor = 'grey', linecolor = 'grey',
        row = 1, col = 1)
    return fig

def _ibi_subplot(
    ibi_x: np.ndarray,
    ibi_y: np.ndarray,
    fig: go.Figure,
    line_dict: dict = dict(color = '#eb4034', width = 1.5),
    name: str = 'IBI'
) -> go.Figure:

    # Get the last row id from the figure
    try:
        grid = fig._grid_ref
        last_row = len(grid) if isinstance(grid, (list, tuple)) else \
        grid.shape[0]
    except Exception:
        last_row = int(getattr(getattr(fig.layout, 'grid', None), 'rows', 1))

    fig.add_trace(
        go.Scatter(
            x = ibi_x, y = ibi_y, name = name,
            line = line_dict,
            connectgaps = True,
            mode = 'lines',
            hovertemplate = f'<b>{name}</b>: %{y:.2f} ms <extra></extra>'
        ),
        row = last_row, col = 1
    )
    fig.update_yaxes(
        title_text = 'ms', title_standoff = 1,
        showgrid = True, gridwidth = 0.5, gridcolor = 'lightgrey',
        griddash = 'dot', tickcolor = 'grey', linecolor = 'grey',
        row = last_row, col = 1
    )
    return fig