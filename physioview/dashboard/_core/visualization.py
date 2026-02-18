"""
Visualization utilities for the PhysioView Dashboard.

This module provides functions for generating table and figure components
for the dashboard UI.

All functions in this module are intended for internal use by the dashboard
and should not be called directly from external code.
"""

from typing import Optional, Tuple
from dash import html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def _cardiac_summary_table(sqa_df: pd.DataFrame) -> Tuple[dbc.Table, list]:
    """Display the cardiac SQA summary table."""
    valid_df = sqa_df[sqa_df.Invalid != 1].copy().reset_index(drop = True)
    # valid_ix = np.where(np.diff(valid_df['N Detected']) < 10)[0]
    # valid_df = valid_df.loc[valid_ix].reset_index(drop = True)
    if valid_df.empty:
        avg_n = 'N/A'
    else:
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
) -> Tuple[dbc.Table, list]:
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