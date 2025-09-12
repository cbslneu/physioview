import numpy as np
import pandas as pd

def convert_acc(signal, sensitivity, fs):
    """Convert an acceleration signal from G-force to units of m/s^2.

    Parameters
    ----------
    signal: array_like
        An array containing the acceleration signal.
    sensitivity : int
        The G-force sensitivity level of the accelerometer
        (e.g., `2` for \u00B1 2g).
    fs : int
        The sampling rate of the accelerometer recording.

    Returns
    -------
    ms2 : array_like
        An array containing the converted acceleration signal.
    """
    ms2 = signal.copy()
    ms2 = (ms2 / (sensitivity * fs)) * 9.81
    return ms2

def compute_magnitude(x, y, z):
    """Compute the magnitude of a 3-axis accelerometer signal.

    Parameters
    ----------
    x : array_like
        An array containing the x-axis accelerometer data.
    y : array_like
        An array containing the y-axis accelerometer data.
    z : array_like
        An array containing the z-axis accelerometer data.

    Returns
    -------
    magnitude : pandas.Series
        A series of acceleration magnitude values.
    """
    axes = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    magnitude = np.sqrt(axes.apply(lambda n: n ** 2).sum(axis = 1))
    return magnitude

def compute_auc(df, signal, fs, seg_size = 60, ts_col = None, norm = None,
                rolling_window = None, rolling_step = 15):
    """Compute the area-under-the-curve of acceleration magnitude
    across a given window size using Riemann sums.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing acceleration magnitude values.
    signal : str
        The name of the column containing the acceleration magnitude.
    fs : int, float
        The sampling rate.
    seg_size : int
        The segment size (in seconds) by which the areas are computed; by
        default, 60.
    ts_col : str, optional
        The name of the column containing timestamps.
    norm: str, optional
        The method with which to normalize the magnitude values; by default,
        None. Possible options include: 'minmax' or 'zscore'.
    rolling_window : int, optional
        The size, in seconds, of the sliding window across which to
        compute the AUC values; by default, None.
    rolling_step : int, optional
        The step size, in seconds, of the sliding windows; by default, 15.

    Returns
    -------
    auc : pandas.DataFrame
        A DataFrame containing the AUC, its normalized value of the
        acceleration magnitude, and corresponding timestamp for each window.

    Notes
    -----
    If a value is given in the `rolling_window` parameter, the rolling
    window approach will override the segmented approach, ignoring any
    `seg_size` value.
    """
    # Get second-by-second AUC values
    seconds_auc = df[signal].groupby(df.index // fs).sum()

    auc = pd.DataFrame()

    # Rolling windows approach
    if rolling_window is not None:
        w = 1
        for n in range(0, len(seconds_auc), rolling_step):
            window_auc = seconds_auc.iloc[n:(n + rolling_window)]

            # Compute the rolling window's normalized AUC if a normalization
            # method is specified
            norm_agg = np.nan
            if norm is not None:
                normalized = []
                if norm not in (None, 'minmax', 'zscore'):
                    raise ValueError('The `norm` parameter must take None, '
                                     '\'minmax\', or \'zscore\'.')
                elif norm == 'minmax':
                    norm_range = window_auc.max() - window_auc.min()
                    for i in window_auc:
                        norm_val = (i - window_auc.min()) / norm_range
                        normalized.append(norm_val)
                else:
                    sd = window_auc.std()
                    for i in window_auc:
                        norm_val = (i - window_auc.mean()) / sd
                        normalized.append(norm_val)

                # Calculate the rolling window's aggregate normalized AUC
                norm_agg = sum(normalized)

            # Calculate the rolling window's aggregate AUC value
            auc_val = window_auc.sum()

            # Get timestamps if a timestamp column is given
            if ts_col is not None:
                ts_auc = df[ts_col].groupby(df.index // fs).first()
                window_ts = ts_auc.iloc[n:(n + rolling_window)].reset_index(
                    drop = True)
                ts = window_ts.iloc[0]
                auc = pd.concat([auc, pd.DataFrame.from_records([{
                    'Moving Window': w,
                    'Timestamp': ts,
                    'AUC': auc_val,
                    'AUC_Norm': norm_agg
                }])], ignore_index = True)
            else:
                auc = pd.concat([auc, pd.DataFrame.from_records([{
                    'Moving Window': w,
                    'AUC': auc_val,
                    'AUC_Norm': norm_agg
                }])], ignore_index = True)
            w += 1
    else:
        # Segment-wise approach
        s = 1
        for n in range(0, len(seconds_auc), seg_size):
            segment_auc = seconds_auc.iloc[n:(n + seg_size)].reset_index(
                drop = True)

            # Compute the segment's normalized AUC if a normalization method
            # is specified
            norm_agg = np.nan
            if norm is not None:
                normalized = []
                if norm not in (None, 'minmax', 'zscore'):
                    raise ValueError('The `norm` parameter must take None, '
                                     '\'minmax\', or \'zscore\'.')
                elif norm == 'minmax':
                    norm_range = segment_auc.max() - segment_auc.min()
                    for i in segment_auc:
                        norm_val = (i - segment_auc.min()) / norm_range
                        normalized.append(norm_val)
                else:
                    sd = segment_auc.std()
                    for i in segment_auc:
                        norm_val = (i - segment_auc.mean()) / sd
                        normalized.append(norm_val)

                # Calculate the segment's aggregate normalized AUC
                norm_agg = sum(normalized)

            # Calculate the segment's aggregate AUC
            auc_val = segment_auc.sum()

            # Get timestamps if a timestamp column is given
            if ts_col is not None:
                ts_auc = df[ts_col].groupby(df.index // fs).first()
                window_ts = ts_auc.iloc[n:(n + seg_size)].reset_index(
                    drop = True)
                ts = window_ts.iloc[0]
                auc = pd.concat([auc, pd.DataFrame.from_records([{
                    'Segment': s,
                    'Timestamp': ts,
                    'AUC': auc_val,
                    'AUC_Norm': norm_agg
                }])], ignore_index = True)
            else:
                auc = pd.concat([auc, pd.DataFrame.from_records([{
                    'Segment': s,
                    'AUC': auc_val,
                    'AUC_Norm': norm_agg
                }])], ignore_index = True)
            s += 1

    # Remove normalized AUC column if all NaN
    if auc['AUC_Norm'].isna().sum() == len(auc):
        auc.drop(['AUC_Norm'], axis = 1, inplace = True)

    return auc

    # auc = pd.DataFrame()
    # normalized = []
    # if rolling_window is not None:
    #     w = 1
    #     for n in range(0, len(seconds_auc), rolling_step):
    #         window_auc = seconds_auc.iloc[n:(n + rolling_window)]
    #         window_auc.reset_index(drop = True, inplace = True)
    #         auc_val = window_auc.sum()
    #
    #         if norm is not None:
    #             normalized.append(norm_val)
    #         if ts_col is not None:
    #             ts = window_auc[ts_col].iloc[0]
    #             auc = pd.concat([auc, pd.DataFrame.from_records([{
    #                 'Moving Window': w,
    #                 'Timestamp': ts,
    #                 'AUC': auc_val
    #             }])], ignore_index = True)
    #         else:
    #             auc = pd.concat([auc, pd.DataFrame.from_records([{
    #                 'Moving Window': w,
    #                 'AUC': auc_val
    #             }])], ignore_index = True)
    #         w += 1
    # else:
    #     s = 1
    #     for n in range(0, len(df), int(fs * seg_size)):
    #         segment_auc = df.iloc[n:(n + int(fs * seg_size))]
    #         segment_auc.reset_index(drop = True, inplace = True)
    #         magnitude_vals = segment_auc[signal]
    #         auc_val, norm_val = _calculate_auc(magnitude_vals, norm)
    #         if norm is not None:
    #             normalized.append(norm_val)
    #         if ts_col is not None:
    #             ts = segment_auc[ts_col].iloc[0]
    #             auc = pd.concat([auc, pd.DataFrame.from_records([{
    #                 'Segment': s,
    #                 'Timestamp': ts,
    #                 'AUC': auc_val
    #             }])], ignore_index = True)
    #         else:
    #             auc = pd.concat([auc, pd.DataFrame.from_records([{
    #                 'Segment': s,
    #                 'AUC': auc_val
    #             }])], ignore_index = True)
    #         s += 1
    # if normalized:
    #     auc['AUC_Norm'] = normalized
    # return auc

def _calculate_auc(magnitude, norm = None):
    """Calculate the area-under-the-curve of acceleration magnitude and its
    normalized value. If not `None`, possible options for `norm` are
    'minmax' or 'zscore'."""
    auc_val = magnitude.sum()
    if norm == 'minmax':
        range_val = magnitude.max() - magnitude.min()
        norm_val = (auc_val - magnitude.min()) / range_val
    elif norm == 'zscore':
        sd = magnitude.std()
        norm_val = (auc_val - magnitude.mean()) / sd
    else:
        norm_val = np.nan
    return auc_val, norm_val