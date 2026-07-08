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

def _cardiac_summary_table(
    sqa_df: pd.DataFrame,
    duration: float,
    windowed: bool = True,
    window_size: int = 60
) -> Tuple[dbc.Table, list]:
    """Display the cardiac SQA summary table."""
    valid_df = sqa_df[sqa_df.Invalid != 1].copy().reset_index(drop = True)
    n_segments = sqa_df['Segment'].max()
    invalid_n = len(sqa_df.loc[sqa_df['Invalid'] == 1])
    invalid_prop = f'{(invalid_n / n_segments) * 100:.2f}%'
    avg_missing = sqa_df['% Missing'].mean()
    avg_missing = f'{0 if pd.isna(avg_missing) else avg_missing:.2f}%'
    avg_artifact = sqa_df['% Artifact'].mean()
    avg_artifact = f'{0 if pd.isna(avg_artifact) else avg_artifact:.2f}%'

    if valid_df.empty:
        avg_hr = 'N/A'
        avg_beats = 'N/A'
    else:
        avg_beats = valid_df['N Detected'].mean()
        if windowed:
            avg_hr = f'{(avg_beats / window_size) * 60:.2f}'
        else:
            total_beats = valid_df['N Detected'].sum()
            avg_hr = f'{(total_beats / duration) * 60:.2f}'

    # Build summary table data
    if not windowed:
        missing_n = sqa_df['N Missing'].mean()
        artifact_n = sqa_df['N Artifact'].mean()
        data = [
            ('Average Heart Rate (bpm)', f'{avg_hr}'),
            ('Average N Detected Beats',
             f'{avg_beats:.2f}' if isinstance(avg_beats, (int, float))
             else f'{avg_beats}'),
            ('Average N Missing Beats', f'{missing_n:.2f}'),
            ('Average N Artifactual Beats', f'{artifact_n:.2f}'),
            ('% Invalid Segments', invalid_prop),
            ('Average % Missing Beats', avg_missing),
            ('Average % Artifactual Beats', avg_artifact),
        ]
    else:
        missing_n = len(sqa_df.loc[sqa_df['N Missing'] > 0])
        artifact_n = len(sqa_df.loc[sqa_df['N Artifact'] > 0])
        data = [
            ('Average Heart Rate (bpm)', f'{avg_hr}'),
            ('Segments with Missing Beats', missing_n),
            ('Segments with Artifactual Beats', artifact_n),
            ('Segments with Invalid Beats', invalid_n),
            ('% Invalid Segments', invalid_prop),
            ('Average % Missing Beats/Segment', avg_missing),
            ('Average % Artifactual Beats/Segment', avg_artifact),
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
    tonic_scl: np.ndarray
) -> Tuple[dbc.Table, list]:
    """Display the EDA SQA summary table."""
    # SCR peaks are already counted per segment during signal quality
    # assessment, using the same segmentation as the rest of the workflow.
    if 'N SCRs' in sqa_df.columns:
        avg_scr_seg = round(float(sqa_df['N SCRs'].mean()), 2)
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
            'Average Heart Rate (bpm)',
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