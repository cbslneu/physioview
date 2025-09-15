from pathlib import Path
from dash import Dash, DiskcacheManager
from physioview.dashboard.utils import _clear_temp, _make_subdirs, _clear_edits
from physioview.dashboard.layout import layout
from physioview.dashboard.callbacks import get_callbacks
import dash_bootstrap_components as dbc
import diskcache
import warnings
warnings.filterwarnings('ignore')

cache = diskcache.Cache(Path('.') / 'cache')
background_callback_manager = DiskcacheManager(cache)

app = Dash(
    __name__,
    update_title = None,
    external_stylesheets = [dbc.themes.LUX, dbc.icons.FONT_AWESOME],
    background_callback_manager = background_callback_manager
)
app.title = 'PhysioView Dashboard'
app.layout = layout
app.server.name = 'PhysioView Dashboard'
get_callbacks(app)

if __name__ == '__main__':
    _make_subdirs()
    _clear_temp()
    _clear_edits()
    cache.clear()
    app.run_server()