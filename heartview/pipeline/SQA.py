from typing import Literal, Optional, Union
from tqdm import tqdm
from math import ceil
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DEBUGGING = False

# ============================== CARDIOVASCULAR ==============================
class Cardio:
    """
    A class for signal quality assessment on cardiovascular data, including
    electrocardiograph (ECG) or photoplethysmograph (PPG) data.

    Parameters/Attributes
    ---------------------
    fs : int
        The sampling rate of the cardiovascular data.
    """

    def __init__(self, fs: int):
        """
        Initialize the Cardiovascular object.

        Parameters
        ----------
        fs : int
            The sampling rate of the ECG or PPG recording.
        """
        self.fs = int(fs)

    def compute_metrics(
        self,
        data: pd.DataFrame,
        beats_ix: Optional[np.ndarray] = None,
        artifacts_ix: Optional[np.ndarray] = None,
        ts_col: Optional[str] = None,
        seg_size: int = 60,
        min_hr: float = 40,
        rolling_window: Optional[int] = None,
        rolling_step: int = 15,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Compute all SQA metrics for cardiovascular data by segment or
        moving window. Metrics per segment or moving window include numbers
        of detected, expected, missing, and artifactual beats and
        percentages of missing and artifactual beats.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame containing pre-processed ECG or PPG data.
        beats_ix : array_like, optional
            An array containing the indices of detected beats. Required if 
            `data` does not contain a "Beat" column with beat occurrences.
        artifacts_ix : array_like, optional
            An array containing the indices of artifactual beats. Required 
            if `data` does not contain an "Artifact" column with artifactual 
            beat occurrences.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        seg_size : int, optional
            The segment size in seconds; by default, 60.
        min_hr : float, optional
            The minimum heart rate against which the number of detected beats
            is considered valid; by default, 40.
        rolling_window : int, optional
            The size, in seconds, of the sliding window across which to
            compute the SQA metrics; by default, None.
        rolling_step : int, optional
            The step size, in seconds, of the sliding windows; by default, 15.
        show_progress : bool, optional
            Whether to display a progress bar while the function runs; by
            default, True.

        Returns
        -------
        metrics : pandas.DataFrame
            A DataFrame with all computed SQA metrics per segment.

        Notes
        -----
        If a value is given in the `rolling_window` parameter, the rolling
        window approach will override the segmented approach, ignoring any
        `seg_size` value.

        Examples
        --------
        >>> from heartview.pipeline import SQA
        >>> sqa = SQA.Cardio(fs = 1000)
        >>> artifacts_ix = sqa.identify_artifacts(beats_ix, method = 'cbd')
        >>> cardio_qa = sqa.compute_metrics(ecg, beats_ix, artifacts_ix, \
        ...                                 ts_col = 'Timestamp', \
        ...                                 seg_size = 60, min_hr = 40)
        """
        from heartview.heartview import compute_ibis

        df = data.copy()
        df.index = df.index.astype(int)

        # Ensure a "Beat" column exists
        if 'Beat' not in df.columns:
            df.loc[beats_ix, 'Beat'] = 1

        # Compute IBIs if no "IBI" column
        if 'IBI' not in df.columns:
            ibi = compute_ibis(df, self.fs, beats_ix, ts_col)
            df['IBI'] = ibi['IBI']

        # Compute SQA metrics across rolling windows
        if rolling_window is not None:
            results = []
            last_valid_hr = np.nan

            # Compute artifacts
            artifacts = self.get_artifacts(
                df, beats_ix, artifacts_ix, seg_size = 1, ts_col = ts_col)
            
            for s, start in enumerate(
                    tqdm(range(0, len(df), rolling_step * self.fs),
                         disable = not show_progress), start = 1):
                window = df.iloc[start: start + rolling_window * self.fs]

                # Calculate expected HR
                median_hrs = self._window_medians(window)
                if median_hrs:
                    exp_hr = float(np.nanmedian(median_hrs))
                    last_valid_hr = exp_hr
                elif not np.isnan(last_valid_hr):
                    exp_hr = last_valid_hr
                else:
                    exp_hr = np.nan

                # Calculate the expected number of beats for this window
                if np.isnan(exp_hr):
                    n_expected = np.nan
                else:
                    n_expected = int(round(exp_hr * (rolling_window / 60.0)))

                # Detected beats in this window
                n_detected = window['Beat'].notna().sum()

                # Missing beats
                n_missing = np.nan if np.isnan(n_expected) \
                    else max(0, n_expected - n_detected)
                perc_missing = np.nan if np.isnan(n_expected) \
                    else round((n_missing / n_expected) * 100, 2)

                # Artifactual beats in this window
                start_sec = start // self.fs
                window_artifact = artifacts.iloc[
                                  start_sec: start_sec + rolling_window]
                n_artifact = window_artifact['N Artifact'].sum()
                perc_artifact = np.nan if n_detected == 0 \
                    else round((n_artifact / n_detected) * 100, 2)

                row = {
                    'Moving Window': s
                }
                if ts_col is not None and ts_col in window.columns:
                    row['Timestamp'] = window[ts_col].iloc[0]
                row.update({
                    'N Expected': n_expected,
                    'N Detected': n_detected,
                    'N Missing': n_missing,
                    '% Missing': perc_missing,
                    'N Artifact': n_artifact,
                    '% Artifact': perc_artifact,
                })
                results.append(row)
            metrics = pd.DataFrame(results)

        # Compute SQA metrics across non-overlapping segments
        else:
            if ts_col is not None:
                missing = self.get_missing(
                    df, beats_ix, artifacts_ix, seg_size, ts_col = ts_col)
                artifacts = self.get_artifacts(
                    df, beats_ix, artifacts_ix, seg_size, ts_col)
                metrics = pd.merge(
                    missing, artifacts, on = ['Segment', 'Timestamp'])
            else:
                missing = self.get_missing(
                    df, beats_ix, artifacts_ix, seg_size)
                artifacts = self.get_artifacts(
                    df, beats_ix, artifacts_ix, seg_size)
                metrics = pd.merge(missing, artifacts, on = ['Segment'])

        metrics['Invalid'] = metrics['N Detected'].apply(
            lambda x: 1 if x < int(min_hr * (seg_size/60)) or x > 220
            else np.nan)

        return metrics

    def get_artifacts(
        self,
        data: pd.DataFrame,
        beats_ix: np.ndarray,
        artifacts_ix: np.ndarray,
        seg_size: int = 60,
        ts_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Summarize the number and proportion of artifactual beats per segment.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame containing the pre-processed ECG or PPG data.
        beats_ix : array_like
            An array containing the indices of detected beats.
        artifacts_ix : array_like
            An array containing the indices of artifactual beats. This is
            outputted from `SQA.Cardio.identify_artifacts()`.
        seg_size : int
            The size of the segment in seconds; by default, 60.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.

        Returns
        -------
        artifacts : pandas.DataFrame
            A DataFrame with the number and proportion of artifactual beats
            per segment.

        See Also
        --------
        SQA.Cardio.identify_artifacts :
            Identify artifactual beats using both or either of the methods.
        """
        df = data.copy()
        if 'Beat' not in df.columns:
            df.loc[beats_ix, 'Beat'] = 1
        if 'Artifact' not in df.columns:
            df.loc[artifacts_ix, 'Artifact'] = 1

        n_seg = ceil(len(df) / (self.fs * seg_size))
        segments = pd.Series(np.arange(1, n_seg + 1))
        n_detected = df.groupby(
            df.index // (self.fs * seg_size))['Beat'].sum().fillna(0).astype(int)
        n_artifact = df.groupby(
            df.index // (self.fs * seg_size))['Artifact'].sum().fillna(0).astype(int)
        perc_artifact = round((n_artifact / n_detected) * 100, 2)

        if ts_col is not None:
            timestamps = df.groupby(
                df.index // (self.fs * seg_size)).first()[ts_col]
            artifacts = pd.concat([
                segments,
                timestamps,
                n_artifact,
                perc_artifact,
            ], axis = 1)
            artifacts.columns = [
                'Segment',
                'Timestamp',
                'N Artifact',
                '% Artifact',
            ]
        else:
            artifacts = pd.concat([
                segments,
                n_artifact,
                perc_artifact,
            ], axis = 1)
            artifacts.columns = [
                'Segment',
                'N Artifact',
                '% Artifact',
            ]
        return artifacts

    def identify_artifacts(
        self,
        beats_ix: np.ndarray,
        method: Literal['hegarty', 'cbd', 'both'],
        initial_hr: Union[float, Literal['auto'], None] = None,
        prev_n: Optional[int] = None,
        neighbors: Optional[int] = None,
        tol: Optional[float] = None
    ) -> np.ndarray:
        """
        Identify locations of artifactual beats in cardiovascular data based
        on the criterion beat difference approach by Berntson et al. (1990),
        the Hegarty-Craver et al. (2018) approach, or both.

        Parameters
        ----------
        beats_ix : array_like
            An array containing the indices of detected beats.
        method : {'hegarty', 'cbd', 'both'}
            The artifact identification method for identifying artifacts.
            This must be 'hegarty', 'cbd', or 'both'.
        initial_hr : float, or 'auto', optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, 'auto' for automatic calculation
            using the mean heart rate value obtained from six consecutive
            IBIs with the smallest average successive difference. Required
            for the 'hegarty' method.
        prev_n : int, optional
            The number of preceding IBIs to validate against; by default, 6.
            Required for 'hegarty' method.
        neighbors : int, optional
            The number of surrounding IBIs with which to derive the criterion
            beat difference score; by default, 5. Required for 'cbd' method.
        tol : float, optional
            A configurable hyperparameter used to fine-tune the stringency of
            the criterion beat difference test; by default, 1. Required for
            'cbd' method.

        Returns
        -------
        artifacts_ix : array_like
            An array containing the indices of identified artifact beats.

        Notes
        -----
        The source code for the criterion beat difference test is from work by
        Hoemann et al. (2020).

        References
        ----------
        Berntson, G., Quigley, K., Jang, J., Boysen, S. (1990). An approach to
        artifact identification: Application to heart period data.
        Psychophysiology, 27(5), 586–598.

        Hegarty-Craver, M. et al. (2018). Automated respiratory sinus
        arrhythmia measurement: Demonstration using executive function
        assessment. Behavioral Research Methods, 50, 1816–1823.

        Hoemann, K. et al. (2020). Context-aware experience sampling reveals
        the scale of variation in affective experience. Scientific
        Reports, 10(1), 1–16.
        """

        def identify_artifacts_hegarty(
            beats_ix: np.ndarray,
            initial_hr: Union[float, Literal['auto']] = 'auto',
            prev_n: int = 6
        ) -> np.ndarray:
            """Identify locations of artifactual beats in cardiovascular data
            based on the approach by Hegarty-Craver et al. (2018)."""

            ibis = (np.diff(beats_ix) / self.fs) * 1000
            beats = beats_ix[1:]  # drop the first beat
            artifact_beats = []
            valid_beats = [beats_ix[0]]  # assume first beat is valid

            # Set the initial IBI to compare against
            if initial_hr == 'auto':
                successive_diff = np.abs(np.diff(ibis))
                min_diff_ix = np.convolve(
                    successive_diff, np.ones(6) / 6, mode = 'valid').argmin()
                first_ibi = ibis[min_diff_ix:min_diff_ix + 6].mean()
            else:
                first_ibi = 60000 / initial_hr

            for n in range(len(ibis)):
                current_ibi = ibis[n]
                current_beat = beats[n]

                # Check against an estimate of the first N IBIs
                if n < prev_n:
                    if n == 0:
                        ibi_estimate = first_ibi
                    else:
                        next_five = np.insert(ibis[:n], 0, first_ibi)
                        ibi_estimate = np.median(next_five)

                # Check against an estimate of the preceding N IBIs
                else:
                    ibi_estimate = np.median(ibis[n - (prev_n):n])

                # Set the acceptable/valid range of IBIs
                low = (26 / 32) * ibi_estimate
                high = (44 / 32) * ibi_estimate

                if low <= current_ibi <= high:
                    valid_beats.append(current_beat)
                else:
                    artifact_beats.append(current_beat)

            return np.array(valid_beats), np.array(artifact_beats)

        def identify_artifacts_cbd(
            beats_ix: np.ndarray,
            neighbors: int = 5,
            tol: float = 1
        ) -> np.ndarray:
            """Identify locations of abnormal interbeat intervals (IBIs) using
             the criterion beat difference test by Berntson et al. (1990)."""

            # Derive IBIs from beat indices
            ibis = ((np.ediff1d(beats_ix)) / self.fs) * 1000

            # Compute consecutive absolute differences across IBIs
            ibi_diffs = np.abs(np.ediff1d(ibis))

            # Initialize an array to store "bad" IBIs
            ibi_bad = np.zeros(shape = len(ibis))
            artifact_beats = []

            if len(ibi_diffs) < neighbors:
                neighbors = len(ibi_diffs)

            for ii in range(len(ibi_diffs)):

                # If there are not enough neighbors in the beginning
                if ii < int(neighbors / 2) + 1:
                    select = np.concatenate(
                        (ibi_diffs[:ii], ibi_diffs[(ii + 1):(neighbors + 1)]))
                    select_ibi = np.concatenate(
                        (ibis[:ii], ibis[(ii + 1):(neighbors + 1)]))

                # If there are not enough neighbors at the end
                elif (len(ibi_diffs) - ii) < (int(neighbors / 2) + 1) and (
                        len(ibi_diffs) - ii) > 1:
                    select = np.concatenate(
                        (ibi_diffs[-(neighbors - 1):ii], ibi_diffs[ii + 1:]))
                    select_ibi = np.concatenate(
                        (ibis[-(neighbors - 1):ii], ibis[ii + 1:]))

                # If there is only one neighbor left to check against
                elif len(ibi_diffs) - ii == 1:
                    select = ibi_diffs[-(neighbors - 1):-1]
                    select_ibi = ibis[-(neighbors - 1):-1]

                else:
                    select = np.concatenate(
                        (ibi_diffs[ii - int(neighbors / 2):ii],
                         ibi_diffs[(ii + 1):(ii + 1 + int(neighbors / 2))]))
                    select_ibi = np.concatenate(
                        (ibis[ii - int(neighbors / 2):ii],
                         ibis[(ii + 1):(ii + 1 + int(neighbors / 2))]))

                # Calculate the quartile deviation
                QD = self._quartile_deviation(select)

                # Calculate the maximum expected difference (MED)
                MED = 3.32 * QD

                # Calculate the minimal artifact difference (MAD)
                MAD = (np.median(select_ibi) - 2.9 * QD) / 3

                # Calculate the criterion beat difference score
                criterion_beat_diff = (MED + MAD) / 2

                # Find indices of IBIs that fail the CBD check
                if (ibi_diffs[ii]) > tol * criterion_beat_diff:

                    bad_neighbors = int(neighbors * 0.25)
                    if ii + (bad_neighbors - 1) < len(beats_ix):
                        artifact_beats.append(
                            beats_ix[ii + 1:(ii + bad_neighbors + 1)])
                    else:
                        artifact_beats.append(
                            beats_ix[ii + 1:(ii + (bad_neighbors - 1))])
                    ibi_bad[ii + 1] = 1

            artifact_beats = np.array(artifact_beats).flatten()
            return artifact_beats

        if method == 'hegarty':
            initial_hr = initial_hr if initial_hr is not None else 'auto'
            prev_n = prev_n if prev_n is not None else 6
            _, artifacts_ix = identify_artifacts_hegarty(
                beats_ix, initial_hr, prev_n)
        elif method == 'cbd':
            neighbors = neighbors if neighbors is not None else 5
            tol = tol if tol is not None else 1
            artifacts_ix = identify_artifacts_cbd(
                beats_ix, neighbors, tol)
        elif method == 'both':
            initial_hr = initial_hr if initial_hr is not None else 'auto'
            prev_n = prev_n if prev_n is not None else 6
            neighbors = neighbors if neighbors is not None else 5
            tol = tol if tol is not None else 1
            _, artifact_hegarty = identify_artifacts_hegarty(
                beats_ix, initial_hr, prev_n)
            artifact_cbd = identify_artifacts_cbd(
                beats_ix, neighbors, tol)
            artifacts_ix = np.union1d(artifact_hegarty, artifact_cbd)
        else:
            raise ValueError(
                'Invalid method. Method must be \'hegarty\', \'cbd\', '
                'or \'both\'.')
        return artifacts_ix

    def get_missing(
        self, 
        data: pd.DataFrame,
        beats_ix: Optional[np.ndarray] = None,
        artifacts_ix: Optional[np.ndarray] = None,
        seg_size: int = 60,
        ts_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Summarize the number and proportion of missing beats per segment.

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame containing the pre-processed ECG or PPG data. 
        beats_ix : array_like, optional
            An array containing the indices of detected beats. Required if 
            `data` doesn't contain a "Beat" column.
        artifacts_ix : array_like, optional
            An array containing the indices of artifactual beats. Required 
            if `data` doesn't contain an "Artifact" column .
        seg_size : int, optional
            The size of the segment in seconds; by default, 60.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.

        Returns
        -------
        missing : pandas.DataFrame
            A DataFrame with detected, expected, and missing numbers of beats 
            per segment.
        """
        from heartview.heartview import compute_ibis
        data = data.copy()

        # Ensure "Beat" and "Artifact" columns exist
        if 'Beat' not in data.columns:
            data.loc[beats_ix, 'Beat'] = 1
        if 'Artifact' not in data.columns:
            data.loc[artifacts_ix, 'Artifact'] = 1

        # Compute IBIs if no "IBI" column
        if 'IBI' not in data.columns:
            ibi = compute_ibis(data, self.fs, beats_ix, ts_col)
            data['IBI'] = ibi['IBI']

        def _expected_hr(seg: int, seg_nums: np.ndarray) -> float:
            """Estimate the expected HR for a segment with adjacent segment 
            fallback."""
            segment = data.loc[data.Segment == seg]
            median_hrs = self._window_medians(segment)

            # Check the last 50% of the previous segment
            if not median_hrs and (seg - 1) in seg_nums:
                prev = data.loc[data.Segment == seg - 1]
                last_half = prev.iloc[-int(seg_size * 0.5):]
                median_hrs = self._window_medians(last_half)

            # Check the first 50% of the next segment
            if not median_hrs and (seg + 1) in seg_nums:
                nxt = data.loc[data.Segment == seg + 1]
                first_half = nxt.iloc[:int(seg_size * 0.5)]
                median_hrs = self._window_medians(first_half)

            return float(np.nanmedian(median_hrs)) if median_hrs else np.nan

        seg_nums = data.Segment.unique()
        results = []
        last_valid_hr = np.nan

        for seg in seg_nums:
            exp_hr = _expected_hr(seg, seg_nums)

            # Detected beats
            n_detected = data.loc[(data.Segment == seg) & data.Beat.notna()].shape[0]

            # If exp_hr cannot be estimated and if there are detected beats,
            # use the last valid exp_hr
            if np.isnan(exp_hr) and not np.isnan(last_valid_hr) and n_detected > 0:
                exp_hr = last_valid_hr
            elif not np.isnan(exp_hr):
                last_valid_hr = exp_hr

            # Calculate expected number of beats in this segment
            if np.isnan(exp_hr):
                n_expected = np.nan
            else:
                n_expected = int(round(exp_hr * (seg_size / 60)))

            # Rescale n_expected for the last partial segment
            if seg == seg_nums[-1]:
                factor = (len(data[data.Segment == seg]) / self.fs) / seg_size
                n_expected = int(round(n_expected * factor))

            # Missing beats
            n_missing = np.nan if np.isnan(n_expected) \
                else max(0, n_expected - n_detected)
            perc_missing = np.nan if np.isnan(n_expected) \
                else round((n_missing / n_expected) * 100, 2)

            row = {'Segment': seg}
            if ts_col is not None and ts_col in data.columns:
                row['Timestamp'] = data.loc[data.Segment == seg, ts_col].iloc[0]
            row.update({
                'N Detected': n_detected,
                'N Expected': n_expected,
                'N Missing': n_missing,
                '% Missing': perc_missing,
            })
            results.append(row)
        missing = pd.DataFrame(results)
        
        # Backfill for any un-estimable leading segments
        first_valid = missing['N Expected'].first_valid_index()
        if first_valid is not None:
            missing.loc[:first_valid, 'N Expected'] = missing.loc[first_valid, 'N Expected']

        # Recalculate missing numbers
        missing['N Expected'] = missing['N Expected'].astype('Int64')
        missing['N Missing'] = (missing['N Expected'] - missing['N Detected']).clip(lower = 0)
        missing['% Missing'] = ((missing['N Missing'] / missing['N Expected']) * 100).round(2)
        return missing

    def get_seconds(
        self,
        data: pd.DataFrame,
        beats_ix: np.ndarray,
        ts_col: Optional[str] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """Get instantaneous (second-by-second) HR, IBI, and beat counts from
        ECG or PPG data according to the approach by Graham (1978).

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame containing the pre-processed ECG or PPG data.
        beats_ix : array-like
            An array containing the indices of detected beats.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        show_progress : bool, optional
            Whether to display a progress bar while the function runs; by
            default, True.

        Returns
        -------
        interval_data : pandas.DataFrame
            A DataFrame containing instantaneous HR and IBI values.

        Notes
        -----
        Rows with `NaN` values in the resulting DataFrame `interval_data`
        denote seconds during which no beats in the data were detected.

        References
        ----------
        Graham, F. K. (1978). Constraints on measuring heart rate and period
        sequentially through real and cardiac time. Psychophysiology, 15(5),
        492–495.
        """
        df = data.copy()
        temp_beat = '_temp_beat'
        df.index = df.index.astype(int)
        df.loc[beats_ix, temp_beat] = 1

        interval_data = []

        # Iterate over each second
        s = 1
        for i in tqdm(range(0, len(df), self.fs), disable = not show_progress):

            # Get data at the current second and evaluation window
            current_sec = df.iloc[i:(i + self.fs)]
            if i == 0:
                # Look at current and next second
                window = df.iloc[:(i + self.fs)]
            else:
                # Look at previous, current, and next second
                window = df.iloc[(i - self.fs):(min(i + self.fs, len(df)))]

            # Get mean IBI and HR values from the detected beats
            current_beats = current_sec[current_sec[temp_beat] == 1].index.values
            window_beats = window[window[temp_beat] == 1].index.values
            ibis = np.diff(window_beats) / self.fs * 1000
            if len(ibis) == 0:
                mean_ibi = np.nan
                mean_hr = np.nan
            else:
                mean_ibi = np.mean(ibis)
                hrs = 60000 / ibis
                r_hrs = 1 / hrs
                mean_hr = 1 / np.mean(r_hrs)

            # Append values for the current second
            if ts_col is not None:
                interval_data.append({
                    'Second': s,
                    'Timestamp': current_sec.iloc[0][ts_col],
                    'Mean HR': mean_hr,
                    'Mean IBI': mean_ibi,
                    'N Beats': len(current_beats)
                })
            else:
                interval_data.append({
                    'Second': s,
                    'Mean HR': mean_hr,
                    'Mean IBI': mean_ibi,
                    'N Beats': len(current_beats)
                })

            s += 1
        interval_data = pd.DataFrame(interval_data)
        return interval_data

    def correct_interval(
        self,
        beats_ix: np.ndarray,
        initial_hr: Union[float, Literal['auto']] = 'auto',
        prev_n: int = 6,
        min_bpm: int = 40,
        max_bpm: int = 200,
        hr_estimate_window: int = 6,
        print_estimated_hr: bool = True,
        short_threshold: float = (24 / 32),
        long_threshold: float = (44 / 32),
        extra_threshold: float = (52 / 32)
    ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Correct artifactual beats in cardiovascular data based on the
        approach by Hegarty-Craver et al. (2018).

        Parameters
        ----------
        beats_ix : array_like
            An array containing the indices of detected beats.
        initial_hr : float or {'auto'}, optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, 'auto' (i.e., the value is
            determined automatically).
        prev_n : int, optional
            The number of preceding IBIs to validate against; by default, 6.
        min_bpm : int, optional
            The minimum possible heart rate in beats per minute (bpm);
            by default, 40.
        max_bpm : int, optional
            The maximum possible heart rate in beats per minute (bpm);
            by default, 200.
        hr_estimate_window : int, optional
            The window size for estimating the heart rate; by default, 6.
        print_estimated_hr : bool, optional
            Whether to print the estimated heart rate; by default, True.
        short_threshold : float, optional
            The threshold for short IBIs; by default, 24/32.
        long_threshold : float, optional
            The threshold for long IBIs; by default, 44/32.
        extra_threshold : float, optional
            The threshold for extra long IBIs; by default, 52/32.

        Returns
        -------
        beats_ix_corrected: array_like
            An array containing the indices of corrected beats.
        corrected_ibis : array_like
            An array containing the indices of corrected IBIs.
        original: pandas.DataFrame
            A DataFrame containing the original IBIs (millisecond-based and
            index-based) and beat indices.
        corrected: pandas.DataFrame
            A DataFrame containing the corrected IBIs (millisecond-based and
            index-based) and beat indices.

        References
        ----------
        Hegarty-Craver, M. et al. (2018). Automated respiratory sinus
        arrhythmia measurement: Demonstration using executive function
        assessment. Behavioral Research Methods, 50, 1816–1823.
        '''
        global MIN_BPM, MAX_BPM
        MIN_BPM = min_bpm
        MAX_BPM = max_bpm

        ibis = np.diff(beats_ix)
        # drop the first beat
        beats = beats_ix[1:]

        global cnt, corrected_ibis, corrected_beats, corrected_flags
        global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, \
            current_flag, correction_flags

        # increment when correcting the ibi and decrement when accepting the ibi
        cnt = 0

        # initialize
        prev_ibi = 0
        prev_beat = 0
        prev_flag = None
        current_ibi = 0
        current_beat = 0
        current_flag = None
        corrected_ibis = []
        corrected_beats = []
        corrected_flags = []
        correction_flags = [0 for i in range(len(beats))]

        # Set the initial IBI to compare against
        global prev_ibis_fifo, first_ibi, correction_failed
        if initial_hr == 'auto':
            successive_diff = np.abs(np.diff(ibis))
            min_diff_ix = np.convolve(
                successive_diff, np.ones(hr_estimate_window) / hr_estimate_window,
                mode = 'valid').argmin()
            first_ibi = ibis[min_diff_ix:min_diff_ix + hr_estimate_window].mean()
            if print_estimated_hr:
                print('Estimated average HR (bpm): ', np.floor(60 / (first_ibi / self.fs)))
        else:
            first_ibi = self.fs * 60 / initial_hr

        # FIFO for the previous n+1 IBIs
        prev_ibis_fifo = self._MaxNFifo(prev_n, first_ibi)
        # Store whether the correction failed for the last n IBIs
        correction_failed = self._MaxNFifo(prev_n - 1)

        def _estimate_ibi(prev_ibis: np.ndarray) -> int:
            '''
            Estimate IBI based on the previous IBIs.

            Parameters
            ----------
            prev_ibis: array_like
                A list of prev_n number of previous IBIs.

            Returns
            -------
            estimated_ibi : int
            '''
            return np.median(prev_ibis)

        def _return_flag(
            current_ibi: int,
            prev_ibis: Optional[np.ndarray] = None
        ) -> str:
            '''
            Return whether the current IBI is correct, short, long, or extra
            long based on the previous IBIs.
                Correct: 26/32 - 44/32 of the estimated IBI
                Short: < 26/32 of the estimated IBI
                Long: > 44/32 and < 54/32 of the estimated IBI
                Extra Long: > 54/32 of the estimated IBI

            Parameters
            ----------
            current_ibi: int
                current IBI value in the number of indices.
            prev_ibis: array_like, optional
                A list of prev_n number of previous IBIs.

            Returns
            -------
            flag : str
                The flag of the current IBI: 'Correct', 'Short', 'Long', or
                'Extra Long'.
            '''

            # Calculate the estimated IBI
            estimated_ibi = _estimate_ibi(prev_ibis)

            # Set the acceptable/valid range of IBIs
            low = short_threshold * estimated_ibi
            high = long_threshold * estimated_ibi
            extra = extra_threshold * estimated_ibi

            # Flag the ibi: correct, short, long, or extra long
            if low <= current_ibi <= high:
                flag = 'Correct'
            elif current_ibi < low:
                flag = 'Short'
            elif current_ibi > high and current_ibi < extra:
                flag = 'Long'
            else:
                flag = 'Extra Long'

            return flag

        def _acceptance_check(
            corrected_ibi: int,
            prev_ibis: np.ndarray
        ) -> bool:
            '''
            Check if the corrected IBI is acceptable (falls within
            27/32 to 42/32 of the estimated IBI).

            Parameters
            ----------
            corrected_ibi: int
                The corrected IBI value.
            prev_ibis: array_like
                A list of prev_n number of previous IBIs.

            Returns
            -------
            bool
                True if the corrected IBI is within the acceptable range,
                False otherwise.
            '''

            # Calculate the estimated IBI
            estimated_ibi = _estimate_ibi(prev_ibis)

            # Set the acceptable/valid range of IBIs
            low = short_threshold * estimated_ibi
            high = long_threshold * estimated_ibi

            # If the corrected value is within the range, return True
            if corrected_ibi >= low and corrected_ibi <= high:
            #if corrected_ibi <= high:
                return True
            else:
                return False

        def _accept_ibi(n: int, correction_failed_flag: int = 0) -> None:
            '''
            Accept the current IBI without correction.

            Parameters
            ----------
            n : int
                The index of the current IBI.
            correction_failed_flag : int, optional
                Flag to indicate whether the correction failed for the
                current IBI; by default, 0. If the flag is 1, the correction
                failed for the current IBI.
            '''
            global prev_ibis_fifo, cnt, correction_failed
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag

            # Check if previous IBI is within limits before accepting current IBI
            _check_limits(n)

            # Fix the previous IBI
            corrected_ibis.append(prev_ibi)
            corrected_beats.append(prev_beat)
            corrected_flags.append(prev_flag)

            # Add the previous IBI to the queue
            prev_ibis_fifo.push(prev_ibi)

            # Update the previous IBI to the current IBI
            prev_ibi = current_ibi
            prev_beat = current_beat
            prev_flag = current_flag

            # Decrement the counter
            cnt = max(0, cnt-1)
            if DEBUGGING:
                print('accepted:', current_ibi,
                      ' flag:', current_flag,
                      ' based on ', prev_ibis_fifo.get_queue()[1:])

            # If the correction failed for the current IBI, push 1 to the
            # correction_failed FIFO, otherwise push 0
            if correction_failed_flag == 0:
                correction_failed.push(0)
            else:
                correction_failed.push(1)

        def _add_prev_and_current(n: int) -> None:
            '''
            Add the previous and current IBIs if the sum is less than 42/32 of
            the estimated IBI.

            Parameters
            ----------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, \
                current_flag, correction_flags

            # Add the previous and current IBIs
            corrected_ibi = prev_ibi + current_ibi

            # Check if the corrected IBI is acceptable
            if _acceptance_check(corrected_ibi, prev_ibis_fifo.get_queue()[1:]):

                # Update the current IBI to the corrected IBI
                current_ibi = corrected_ibi
                current_beat = current_beat
                current_flag = _return_flag(current_ibi, prev_ibis_fifo.get_queue()[1:])
                if n == 1:
                    # Update the previous IBI to the current IBI
                    prev_ibi = current_ibi
                    prev_beat = current_beat
                    prev_flag = current_flag
                else:
                    # Pull up the second previous IBI as previous IBI
                    prev_ibi = corrected_ibis[-1]
                    prev_beat = corrected_beats[-1]
                    prev_flag = corrected_flags[-1]

                    # Check if the previous IBI is within the limits before
                    # accepting the current IBI
                    _check_limits(n)

                    # Check_limits function may update the previous IBI pulled,
                    # so update the value
                    corrected_ibis[-1] = prev_ibi
                    corrected_beats[-1] = prev_beat
                    corrected_flags[-1] = prev_flag

                    # Update the last IBI value in the queue
                    prev_ibis_fifo.change_last(prev_ibi)

                    # Update the previous IBI to the current IBI
                    prev_ibi = current_ibi
                    prev_beat = current_beat
                    prev_flag = current_flag

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING:
                    print('added:', current_ibi,
                          ' flag:', current_flag,
                          ' based on ', prev_ibis_fifo.get_queue()[1:])
            else:
                if DEBUGGING:
                    print('acceptance check failed for adding: ', corrected_ibi)

                # If the corrected IBI is not acceptable, accept the current IBI
                _accept_ibi(n, correction_failed_flag = 1)

        def _add_secondprev_and_prev(n: int) -> None:
            '''
            Add the second previous and previous IBIs if the sum is less than
            42/32 of the estimated IBI.

            Parameters
            ----------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, \
                current_flag, correction_flags

            # Add the previous and current IBIs
            corrected_ibi = corrected_ibis[-1] + prev_ibi

            # Check if the corrected IBI is acceptable
            # Use IBIs before the second previous IBI
            if _acceptance_check(corrected_ibi, prev_ibis_fifo.get_queue()[:-2]):
                # Update the current IBI to the corrected IBI

                # Pull up the second previous IBI as previous IBI
                prev_ibi = corrected_ibi
                prev_beat = prev_beat
                prev_flag = _return_flag(prev_ibi, prev_ibis_fifo.get_queue()[:-2])

                # Check if the previous IBI is within the limits before accepting the current IBI
                _check_limits(n)

                # Update the value
                corrected_ibis[-1] = prev_ibi
                corrected_beats[-1] = prev_beat
                corrected_flags[-1] = prev_flag

                # Update the last IBI value in the queue
                prev_ibis_fifo.change_last(prev_ibi)

                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag

                # Flag that previous and current IBIs are corrected
                correction_flags[n-2] = 1
                correction_flags[n-1] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING:
                    print('added second prev + prev:', prev_ibi,
                          ' flag:', prev_flag,
                          ' based on ', prev_ibis_fifo.get_queue()[:-2])
            else:
                if DEBUGGING:
                    print('acceptance check failed for adding second prev + prev: ', corrected_ibi)

                # If the corrected IBI is not acceptable, accept the current IBI
                _accept_ibi(n, correction_failed_flag = 1)

        def _insert_interval(n: int) -> None:
            '''
            Split the (previous IBI + current IBI) into multiple intervals.
            The number of splits is determined based on the initial_hr parameter.

            Parameters
            ----------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt, first_ibi
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, \
                current_flag, correction_flags

            # Calculate the number of splits
            n_split = round((prev_ibi + current_ibi) / _estimate_ibi(
                prev_ibis_fifo.get_queue()[1:]), 0).astype(int)

            # Calculate the new IBI
            ibi = np.floor((prev_ibi + current_ibi) / n_split)

            # Check if the corrected IBI is acceptable
            if _acceptance_check(ibi, prev_ibis_fifo.get_queue()[1:]):

                # Fix inserted IBIs other than previous/current IBIs
                for i in range(n_split - 2):
                    corrected_ibis.append(ibi)
                    corrected_flags.append(_return_flag(ibi, prev_ibis_fifo.get_queue()[1:]))
                    if (n == 1 and i == 0) | (len(corrected_beats) == 0):
                        corrected_beats.append(beats_ix[0] + ibi)
                    else:
                        corrected_beats.append(corrected_beats[-1] + ibi)

                    # Add to the queue
                    prev_ibis_fifo.push(ibi)

                # Update the previous IBI
                prev_ibi = ibi
                if len(corrected_beats) > 0:
                    prev_beat = corrected_beats[-1] + ibi
                else:
                    prev_beat = beats_ix[0] + ibi
                prev_flag = _return_flag(ibi, prev_ibis_fifo.get_queue()[:-1])

                # Update the current IBI
                current_ibi = current_beat - prev_beat
                current_flag = _return_flag(ibi, prev_ibis_fifo.get_queue()[1:])

                # Check if the previous IBI is within the limits
                _check_limits(n)

                # Fix the previous IBI
                corrected_ibis.append(prev_ibi)
                corrected_beats.append(prev_beat)
                corrected_flags.append(prev_flag)

                # Add to the queue
                prev_ibis_fifo.push(prev_ibi)

                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter by n_split - 1 in this case
                cnt += n_split - 1

                if DEBUGGING:
                    print('inserted ', n_split - 2,
                          ' intervals: ', ibi,
                          ' flag:', current_flag,
                          ' based on ', prev_ibis_fifo.get_queue()[1:])
            else:
                if DEBUGGING:
                    print('acceptance check failed for inserting: ', ibi)

                # If the corrected IBI is not acceptable, accept the current IBI
                _accept_ibi(n, correction_failed_flag = 1)

        def _average_prev_and_current(n: int) -> None:
            '''
            Average the previous and current IBIs.

            Parameters
            ----------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags

            # Average the previous and current IBIs
            ibi = np.floor((prev_ibi + current_ibi) / 2)

            # Check if the corrected IBI is acceptable
            if _acceptance_check(ibi, prev_ibis_fifo.get_queue()[1:]):
                # Update the previous and current IBI
                prev_ibi = ibi
                if n == 1:
                    prev_beat = beats_ix[0] + ibi
                else:
                    prev_beat = corrected_beats[-1] + ibi
                prev_flag = _return_flag(ibi, prev_ibis_fifo.get_queue()[:-1])
                current_ibi = current_beat - prev_beat
                current_flag = _return_flag(ibi, prev_ibis_fifo.get_queue()[1:])

                # Check if the previous IBI is within the limits
                _check_limits(n)

                # Fix the previous IBI
                corrected_ibis.append(prev_ibi)
                corrected_beats.append(prev_beat)
                corrected_flags.append(prev_flag)

                # Add to the queue
                prev_ibis_fifo.push(prev_ibi)

                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING:
                    print('averaged:', ibi, ' flag:', current_flag, ' based on ', prev_ibis_fifo.get_queue()[1:])
            else:
                if DEBUGGING:
                    print('acceptance check failed for averaging: ', ibi)
                _accept_ibi(n, correction_failed_flag=1)

        def _check_limits(n):
            '''
            Check if the previous IBI (n-1) is within the limits.
            If it is longer the maximum IBI, shorten the previous IBI and lengthen the current IBI.
            If it is shorter than the minimum IBI, lengthen the previous IBI and shorten the current IBI.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, \
                current_flag, correction_flags
            MIN_IBI = np.floor(self.fs * 60 / MAX_BPM)         # minimum IBI in indices
            MAX_IBI = np.floor(self.fs * 60 / MIN_BPM)         # maximum IBI in indices

            # If the previous IBI is shorter than the minimum IBI, lengthen the previous IBI and shorten the current IBI
            if prev_ibi < MIN_IBI:
                remainder = MIN_IBI - prev_ibi
                prev_beat = prev_beat + remainder
                prev_ibi = MIN_IBI
                prev_flag = _return_flag(prev_ibi, prev_ibis_fifo.get_queue()[:-1])
                current_ibi = current_ibi - remainder
                current_flag = _return_flag(current_ibi, prev_ibis_fifo.get_queue()[1:])

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING:
                    print('Shorter than the minimum IBI and corrected: ', prev_ibi, ' ', prev_flag, ' | ', current_ibi, ' ', current_flag)

            # If the previous IBI is longer than the maximum IBI, shorten the previous IBI and lengthen the current IBI
            elif prev_ibi > MAX_IBI:
                remainder = prev_ibi - MAX_IBI
                prev_beat = prev_beat - remainder
                prev_ibi = MAX_IBI
                prev_flag = _return_flag(prev_ibi, prev_ibis_fifo.get_queue()[:-1])
                current_ibi = current_ibi + remainder
                current_flag = _return_flag(current_ibi, prev_ibis_fifo.get_queue()[1:])

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING:
                    print('Longer than the maximum IBI and corrected: ', prev_ibi, ' ', prev_flag, ' | ', current_ibi, ' ', current_flag)
            return

        for n in range(len(ibis)):
            current_ibi = ibis[n]
            current_beat = beats[n]

            # Accept the first ibi
            if n == 0:
                current_flag = _return_flag(current_ibi, prev_ibis = prev_ibis_fifo.get_queue())
                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag

            else:
                current_flag = _return_flag(current_ibi, prev_ibis = prev_ibis_fifo.get_queue()[:-1])

                if DEBUGGING:
                    print('n:', n)
                    print('prev:', prev_ibi, ' ', prev_flag,
                          ' | current:', current_ibi, ' ', current_flag)

                # If current IBI is correct
                if current_flag == 'Correct':
                    # If previous IBI is correct/long, accept current
                    if prev_flag == 'Correct' or prev_flag == 'Long':
                        _accept_ibi(n)
                    elif prev_flag == 'Short':
                        if n == 1:
                            _add_prev_and_current(n)
                        else:
                            # If previous IBI is shorter than current IBI, add them together
                            if corrected_ibis[-1] > current_ibi:
                                _add_prev_and_current(n)
                            else:
                                _add_secondprev_and_prev(n)
                    # If previous IBI is extra long, split previous and current
                    elif prev_flag == 'Extra Long':
                        _insert_interval(n)

                # If current IBI is short
                elif current_flag == 'Short':
                    # If previous IBI is correct, accept it
                    if prev_flag == 'Correct':
                        _accept_ibi(n)
                    # If previous IBI is short, add previous + current
                    elif prev_flag == 'Short':
                        _add_prev_and_current(n)
                    # If previous IBI is long/extra long, average previous and current
                    elif prev_flag == 'Long' or prev_flag == 'Extra Long':
                        _average_prev_and_current(n)

                # If the current IBI is long
                elif current_flag == 'Long':
                    # If previous IBI is correct or long, accept it
                    if prev_flag == 'Correct' or prev_flag == 'Long':
                        _accept_ibi(n)
                    # If previous IBI is short, average previous and current
                    elif prev_flag == 'Short':
                        _average_prev_and_current(n)
                    # If previous IBI is extra long, split previous and current
                    elif prev_flag == 'Extra Long':
                        _insert_interval(n)

                # If current IBI is extra long
                elif current_flag == 'Extra Long':
                    # If previous IBI is correct, long, or extra long, split previous and current
                    if prev_flag == 'Correct' or prev_flag == 'Long' or prev_flag == 'Extra Long':
                        _insert_interval(n)
                    # If previous IBI is short, average previous and current
                    elif prev_flag == 'Short':
                        _average_prev_and_current(n)

            # If more than 3 corrections are made in the last prev_n IBIs, reset the FIFO
            if sum(correction_failed.get_queue()) >= 3:
                prev_ibis_fifo.reset(first_ibi)

        # Add the last beat
        corrected_ibis.append(current_ibi)
        corrected_beats.append(current_beat)
        corrected_flags.append(current_flag)

        correction_flags = np.array(correction_flags).astype(int)

        # Convert the IBIs to milliseconds
        original_ibis_ms = np.round((np.array(ibis) / self.fs) * 1000, 2)

        original = pd.DataFrame({
            'Original IBI (ms)': np.insert(original_ibis_ms, 0, np.nan),
            'Original IBI (index)': np.insert(ibis.astype(object), 0, np.nan),
            'Original Beat': np.insert(beats, 0, beats_ix[0]),
            'Correction': np.insert(correction_flags, 0, 0)
        })

        corrected_ibis_ms = np.round((np.array(corrected_ibis) / self.fs) * 1000, 2)

        corrected_ibis = np.array(corrected_ibis).astype(object)
        corrected_flags = np.array(corrected_flags).astype(object)

        # Add the first beat and create a dataframe
        corrected = pd.DataFrame({
            'Corrected IBI (ms)': np.insert(corrected_ibis_ms, 0, np.nan),
            'Corrected IBI (index)': np.insert(corrected_ibis, 0, np.nan),
            'Corrected Beat': np.insert(corrected_beats, 0, beats_ix[0]),
            'Flag': np.insert(corrected_flags, 0, np.nan)
        })
        beats_ix_corrected = np.insert(corrected_beats, 0, beats_ix[0]).astype(int)
        return beats_ix_corrected, corrected_ibis, original, corrected

    def get_corrected(
        self,
        beats_ix: np.ndarray,
        seg_size: int = 60,
        initial_hr: Union[float, Literal['auto']] = 'auto',
        prev_n: int = 6,
        min_bpm: int = 40,
        max_bpm: int = 200,
        hr_estimate_window: int = 6,
        print_estimated_hr: bool = True,
        short_threshold: float = (24 / 32),
        long_threshold: float = (44 / 32),
        extra_threshold: float = (52 / 32)
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the corrected interbeat intervals (IBIs) and beat indices.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame containing the pre-processed ECG or PPG data.
        beats_ix : array_like
            An array containing the indices of detected beats.
        seg_size : int
            The size of the segment in seconds; by default, 60.
        initial_hr : float or {'auto'}, optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, 'auto' (i.e., the value is
            determined automatically).
        prev_n : int, optional
            The number of preceding IBIs to validate against; by default, 6.
        min_bpm : int, optional
            The minimum possible heart rate in beats per minute (bpm);
            by default, 40.
        max_bpm : int, optional
            The maximum possible heart rate in beats per minute (bpm);
            by default, 200.
        hr_estimate_window : int, optional
            The window size for estimating the heart rate; by default, 6.
        print_estimated_hr : bool, optional
            Whether to print the estimated heart rate; by default, True.
        short_threshold : float, optional
            The threshold for short IBIs; by default, 24/32.
        long_threshold : float, optional
            The threshold for long IBIs; by default, 44/32.
        extra_threshold : float, optional
            The threshold for extra long IBIs; by default, 52/32.

        Returns
        -------
        original : pandas.DataFrame
            A data frame containing the original IBIs and beat indices.
        corrected : pandas.DataFrame
            A data frame containing the corrected IBIs and beat indices.
        combined : pandas.DataFrame
            A data frame containing the summary of flags in each segment.
        """

        # Get the corrected IBIs and beat indices
        _, original, corrected = self.correct_interval(
            beats_ix = beats_ix, initial_hr = initial_hr,
            prev_n = prev_n, min_bpm = min_bpm, max_bpm = max_bpm,
            hr_estimate_window = hr_estimate_window,
            print_estimated_hr = print_estimated_hr,
            short_threshold = short_threshold, long_threshold = long_threshold,
            extra_threshold = extra_threshold)

        # Get the segment number for each beat
        for row in original.iterrows():
            seg = ceil(row[1].loc['Original Beat'] / (seg_size * self.fs))
            original.loc[row[0], 'Segment'] = seg
        for row in corrected.iterrows():
            seg = ceil(row[1].loc['Corrected Beat'] / (seg_size * self.fs))
            corrected.loc[row[0], 'Segment'] = seg
        original['Segment'] = original['Segment'].astype(pd.Int64Dtype())
        corrected['Segment'] = corrected['Segment'].astype(pd.Int64Dtype())

        # Get the number and percentage of corrected beats in each segment
        original_seg = original.groupby('Segment')['Correction'].sum().astype(pd.Int64Dtype())
        original_seg = pd.DataFrame(original_seg.reset_index(name = '# Corrected'))
        original_seg_nbeats = original.groupby('Segment')['Correction'].count().astype(pd.Int64Dtype())
        original_seg_nbeats = pd.DataFrame(original_seg_nbeats.reset_index(name = '# Beats'))
        original_seg = original_seg.merge(original_seg_nbeats, on = 'Segment')
        original_seg['% Corrected'] = round((original_seg['# Corrected'] / original_seg['# Beats']) * 100, 2)
        original_seg.drop('# Beats', axis = 1, inplace = True)

        # Get the number of each flag (Correct/Short/Long/Extra Long) in each segment
        corrected_seg = corrected.groupby('Segment')['Flag'].value_counts().astype(pd.Int64Dtype())
        corrected_seg = pd.DataFrame(corrected_seg.reset_index(name = 'Count'))
        corrected_seg = corrected_seg.pivot(index = 'Segment', columns = 'Flag', values = 'Count').reset_index().fillna(0)
        corrected_seg.columns.name = None
        corrected_seg = corrected_seg.rename_axis(None, axis = 1)

        combined = pd.merge(corrected_seg, original_seg, on='Segment')

        return original, corrected, combined

    def plot_missing(
        self,
        sqa_metrics: pd.DataFrame,
        invalid_thresh = 30,
        title = None
    ) -> go.Figure:
        """
        Plot detected and missing beat counts.

        Parameters
        ----------
        sqa_metrics : pandas.DataFrame
            The DataFrame containing SQA metrics per segment.
        invalid_thresh : int, float
            The minimum number of beats detected for a segment to be considered
            valid; by default, 30.
        title : str, optional
            The title of the plot.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A Plotly bar chart of detected and missing beat counts.
        """
        max_beats = ceil(sqa_metrics['N Detected'].max() / 10) * 10
        nearest = ceil(max_beats / 2) * 2
        dtick_value = nearest / 5

        fig = go.Figure(
            data = [
                go.Bar(
                    x = sqa_metrics['Segment'],
                    y = sqa_metrics['N Expected'],
                    name = 'Missing',
                    marker = dict(color = '#f2816d'),
                    hovertemplate = '<b>Segment %{x}:</b> %{customdata:.0f} '
                                    'missing<extra></extra>'),
                go.Bar(
                    x = sqa_metrics['Segment'],
                    y = sqa_metrics['N Detected'],
                    name = 'Detected',
                    marker = dict(color = '#313c42'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                                    'detected<extra></extra>')
            ]
        )
        fig.data[0].update(customdata = sqa_metrics['N Missing'])

        # Get invalid segment data points
        invalid_x = []
        invalid_y = []
        invalid_text = []
        for segment_num, n_detected in zip(
                sqa_metrics['Segment'], sqa_metrics['N Detected']):
            if n_detected < invalid_thresh:
                invalid_x.append(segment_num)
                invalid_y.append(
                    n_detected + 3)
                invalid_text.append('<b>!</b>')

        # Add scatter trace for invalid markers
        if invalid_x:
            fig.add_trace(go.Scatter(
                x = invalid_x,
                y = invalid_y,
                mode = 'text',
                text = invalid_text,
                textposition = 'top center',
                textfont = dict(size = 20, color = '#db0f0f'),
                showlegend = False,
                hoverinfo = 'skip'    # disable tooltips
            ))
        if invalid_x:
            fig.add_annotation(
                text = '<span style="color: #db0f0f"><b>!</b></span>  '
                       'Invalid Number of Beats ',
                align = 'right',
                showarrow = False,
                xref = 'paper',
                yref = 'paper',
                x = 1,
                y = 1.3)

        fig.update_layout(
            xaxis_title = 'Segment Number',
            xaxis = dict(
                tickmode = 'linear',
                dtick = 1,
                range = [sqa_metrics['Segment'].min() - 0.5,
                         sqa_metrics['Segment'].max() + 0.5]),
            yaxis = dict(
                title = 'Number of Beats',
                range = [0, max_beats],
                dtick = dtick_value),
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.0,
                xanchor = 'right',
                x = 1.0),
            font = dict(family = 'Poppins', size = 13),
            height = 289,
            margin = dict(t = 70, r = 20, l = 40, b = 65),
            barmode = 'overlay',
            template = 'simple_white',
        )
        if title is not None:
            fig.update_layout(
                title = title
            )
        return fig

    def plot_artifact(
        self,
        sqa_metrics: pd.DataFrame,
        invalid_thresh: int = 30,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot detected and artifact beat counts.

        Parameters
        ----------
        sqa_metrics : pandas.DataFrame
            The DataFrame containing SQA metrics per segment.
        invalid_thresh : int, float
            The minimum number of beats detected for a segment to be considered
            valid; by default, 30.
        title : str, optional
            The title of the plot.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A Plotly bar chart of detected and missing beat counts.
        """
        max_beats = ceil(sqa_metrics['N Detected'].max() / 10) * 10
        nearest = ceil(max_beats / 2) * 2
        dtick_value = nearest / 5

        fig = go.Figure(
            data = [
                go.Bar(
                    x = sqa_metrics['Segment'],
                    y = sqa_metrics['N Detected'],
                    name = 'Detected',
                    marker = dict(color = '#313c42'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                                    'detected<extra></extra>'),
                go.Bar(
                    x = sqa_metrics['Segment'],
                    y = sqa_metrics['N Artifact'],
                    name = 'Artifact',
                    marker = dict(color = '#f2b463'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                                    'artifact<extra></extra>')
            ],
        )

        # Get invalid segment data points
        invalid_x = []
        invalid_y = []
        invalid_text = []
        for segment_num, n_detected in zip(
                sqa_metrics['Segment'], sqa_metrics['N Detected']):
            if n_detected < invalid_thresh:
                invalid_x.append(segment_num)
                invalid_y.append(
                    n_detected + 3)
                invalid_text.append('<b>!</b>')

        # Add scatter trace for invalid markers
        if invalid_x:
            fig.add_trace(go.Scatter(
                x = invalid_x,
                y = invalid_y,
                mode = 'text',
                text = invalid_text,
                textposition = 'top center',
                textfont = dict(size = 20, color = '#db0f0f'),
                showlegend = False,
                hoverinfo = 'skip'    # disable tooltips
            ))
        if invalid_x:
            fig.add_annotation(
                text = '<span style="color: #db0f0f"><b>!</b></span>  '
                       'Invalid Number of Beats ',
                align = 'right',
                showarrow = False,
                xref = 'paper',
                yref = 'paper',
                x = 1,
                y = 1.3)

        fig.update_layout(
            xaxis_title = 'Segment Number',
            xaxis = dict(
                tickmode = 'linear',
                dtick = 1,
                range = [sqa_metrics['Segment'].min() - 0.5,
                         sqa_metrics['Segment'].max() + 0.5]),
            yaxis = dict(
                title = 'Number of Beats',
                range = [0, max_beats],
                dtick = dtick_value),
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.0,
                xanchor = 'right',
                x = 1.0,
                traceorder = 'reversed'),
            font = dict(family = 'Poppins', size = 13),
            height = 289,
            margin = dict(t = 70, r = 20, l = 40, b = 65),
            barmode = 'overlay',
            template = 'simple_white',
        )
        if title is not None:
            fig.update_layout(
                title = title
            )
        return fig

    def _get_iqr(self, data: np.ndarray) -> float:
        """Compute the interquartile range of a data array."""
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        return iqr

    def _quartile_deviation(self, data: np.ndarray) -> float:
        """Compute the quartile deviation in the criterion beat difference
        test."""
        iqr = self._get_iqr(data)
        QD = iqr * 0.5
        return QD

    def _window_medians(self, segment: pd.DataFrame, win_size: int = 5) -> list:
        """Calculate median HRs from artifact-free windows in a segment 
        slice for get_missing()."""
        median_hrs = []
        beats = segment.dropna(subset = ['Beat'])
        n = len(beats)
        for i in range(n - win_size + 1):
            window = beats.iloc[i:i + win_size]
            if window.Artifact.any():
                continue
            ibi_vals = window.IBI.values
            med_hr = np.nanmedian(60000 / ibi_vals)
            median_hrs.append(med_hr)
        return median_hrs

    class _MaxNFifo:
        """
        A class for FIFO with N elements at maximum.

        Parameters/Attributes
        ---------------------
        prev_n : int
            The maximum number of elements in the FIFO.
        item : int, optional
            The initial item to add to the FIFO; by default, None.
            The item is added twice if it is not None.
        """

        def __init__(self, prev_n: int, item: Optional[int] = None):
            """
            Initialize the FIFO object.

            Parameters
            ----------
            prev_n : int
                The maximum number of elements in the FIFO.
            item : int, optional
                The initial item to add to the FIFO; by default, None.
                The item is added twice if it is not None.
            """
            self.prev_n = prev_n
            if item is not None:
                self.queue = [item, item]
            else:
                self.queue = []

        def push(self, item: int) -> None:
            """
            Push an item to the FIFO. If the number of elements exceeds the maximum, remove the first element.

            Parameters
            ----------
            item : int
                The item to add to the FIFO.
            """
            self.queue.append(item)
            if len(self.queue) > self.prev_n + 1:
                self.queue.pop(0)

        def get_queue(self) -> list:
            """
            Return the FIFO queue.

            Return
            ------
            queue : list
            """
            return self.queue

        def change_last(self, item: int) -> None:
            """
            Change the last item in the FIFO queue.

            Parameters
            ----------
            item : int
                The new item to replace the last item in the queue.
            """
            self.queue[-1] = item

        def reset(self, item: Optional[int] = None) -> None:
            """
            Reset the FIFO queue. If an item is given, reset the queue with
            the item. If not, reset the queue with an empty list.

            Parameters
            ----------
            item: int, optional
                The item to add to the FIFO; by default, None.
                The item is added twice if it is not None.
            """
            if item is None:
                self.queue = []
            else:
                self.queue = [item, item]

# =================================== EDA ====================================
class EDA:
    """
    A class for signal quality assessment on electrodermal activity (EDA)
    data.

    Parameters/Attributes
    ---------------------
    fs : int
        The sampling rate of the EDA data.
    eda_min : float, optional
        The minimum acceptable value for EDA data in microsiemens; by
        default, 0.05 uS.
    eda_max : float, optional
        The maximum acceptable value for EDA data in microsiemens; by
        default, 60 uS.
    eda_max_slope : float, optional
        The maximum slope of EDA data in microsiemens per second; by
        default, 5 uS/sec.
    temp_min : float, optional
        The minimum acceptable temperature in degrees Celsius; by
        default, 20.
    temp_max : float, optional
        The maximum acceptable temperature in degrees Celsius; by
        default, 40.
    invalid_spread_dur : float, optional
        The transition radius for artifacts in seconds; by default, 2.
    """

    def __init__(
        self,
        fs: int,
        eda_min: float = 0.2,
        eda_max: float = 40,
        eda_max_slope: float = 5,
        temp_min: float = 20,
        temp_max: float = 40,
        invalid_spread_dur: float = 2.5
    ):
        """
        Initialize the EDA object.

        Parameters
        ----------
        fs : int
            The sampling rate of the ECG or PPG recording.
        eda_min : float, optional
            The minimum acceptable value for EDA data in microsiemens; by
            default, 0.05 uS.
        eda_max : float, optional
            The maximum acceptable value for EDA data in microsiemens; by
            default, 60 uS.
        eda_max_slope : float, optional
            The maximum slope of EDA data in microsiemens per second; by
            default, 5 uS/sec.
        temp_min : float, optional
            The minimum acceptable temperature in degrees Celsius; by
            default, 20.
        temp_max : float, optional
            The maximum acceptable temperature in degrees Celsius; by
            default, 40.
        invalid_spread_dur : float, optional
            The transition radius for artifacts in seconds; by default,
            2.5 seconds.
        """

        # Check inputs
        if eda_min >= eda_max:
            raise ValueError('`eda_min` must be smaller than `eda_max`.')
        if temp_min >= temp_max:
            raise ValueError('`temp_min` must be smaller than `temp_max`.')

        self.fs = fs
        self.eda_min = eda_min
        self.eda_max = eda_max
        self.eda_max_slope = eda_max_slope
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.invalid_spread_dur = invalid_spread_dur

    def get_validity_metrics(
        self,
        signal: np.ndarray,
        temp: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        preprocessed: bool = True,
    ) -> pd.DataFrame:
        """
        Assess and flag valid and invalid EDA data points.

        Parameters
        ----------
        signal : array_like
            An array containing the EDA signal in microsiemens.
        temp : array_like, optional
            An array containing temperature data in Celsius.
        timestamps : array_like, optional
            An array of timestamps corresponding to each data point.
        preprocessed : bool, optional
            Whether filtered EDA data is being inputted; by default, True.
            If False, an FIR low-pass filter is applied.

        Returns
        -------
        eda_validity : pd.DataFrame
            A DataFrame with the columns:
            - 'Timestamp' (if provided)
            - 'EDA'
            - 'Temp' (if provided)
            - 'Valid' (1 if valid, NaN otherwise)
            - 'Invalid' (1 if invalid, NaN otherwise)
        """
        valid_ix, invalid_ix, _ = self._edaqa(signal, temp, preprocessed)
        eda_validity = pd.DataFrame({
            'Timestamp': timestamps if timestamps is not None else np.arange(
                len(signal)),
            'EDA': signal,
        })
        if temp is not None:
            eda_validity['TEMP'] = temp
        eda_validity.loc[valid_ix, 'Valid'] = 1
        eda_validity.loc[invalid_ix, 'Invalid'] = 1

        return eda_validity

    def get_quality_metrics(
        self,
        signal: np.ndarray,
        temp: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Assess and flag rule violations of EDA quality based on
        the quality assessment procdure by Kleckner et al. (2017).

        Parameters
        ----------
        signal : array_like
            The EDA signal in microsiemens.
        temp : array_like, optional
            Temperature data in Celsius.
        timestamps : array_like, optional
            Array of timestamps corresponding to each data point.

        Returns
        -------
        eda_quality : pd.DataFrame
            A DataFrame with the columns:
            - 'Timestamp'
            - 'EDA'
            - 'Temp' (if provided)
            - 'Out of Range'
            - 'Excessive Slope'
            - 'Temp Out of Range' (if provided)

        References
        ----------
        Kleckner, I.R., Jones, R. M., Wilder-Smith, O., Wormwood, J.B.,
        Akcakaya, M., Quigley, K.S., ... & Goodwin, M.S. (2017). Simple,
        transparent, and flexible automated quality assessment procedures for
        ambulatory electrodermal activity data. IEEE Transactions on Biomedical
        Engineering, 65(7), 1460-1467.
        """
        sampling_interval = 1 / self.fs

        # Rule-specific masks
        mask_out_of_range = self._check_out_of_range(signal)
        mask_excessive_slope = self._check_excessive_slope(
            signal, sampling_interval)
        mask_temp = self._check_temp_out_of_range(
            temp) if temp is not None else None

        # Combine all available checks
        combined_invalid = mask_out_of_range | mask_excessive_slope
        if mask_temp is not None:
            combined_invalid |= mask_temp

        eda_quality = pd.DataFrame({
            'EDA': signal,
            'Out of Range': np.where(mask_out_of_range, 1, np.nan),
            'Excessive Slope': np.where(mask_excessive_slope, 1, np.nan),
        })
        if temp is not None:
            eda_quality['TEMP'] = temp
        if timestamps is not None:
            eda_quality.insert(0, 'Timestamp', timestamps)
        else:
            eda_quality.insert(0, 'Sample', np.arange(len(signal)) + 1)

        if mask_temp is not None:
            eda_quality['Temp Out of Range'] = np.where(mask_temp, 1, np.nan)
        return eda_quality

    def compute_metrics(
        self,
        signal: np.ndarray,
        temp: Optional[np.ndarray] = None,
        preprocessed: bool = True,
        peaks_ix: Optional[np.ndarray] = None,
        seg_size: int = 60,
        rolling_window: Optional[int] = None,
        rolling_step: int = 15,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Assess the quality of electrodermal activity (EDA) data using the rules
        defined by Kleckner et al. (2017). The method identifies valid and
        invalid data points and computes rule-specific quality metrics (e.g.,
        proportions of out-of-range points, excessive slopes, temperature
        violations, and spread-invalid counts), either by segment or across
        sliding windows.

        Parameters
        ----------
        signal : array_like
            An array containing the EDA signal in microsiemens.
        temp : array_like, optional
            An optional array containing temperature data in Celsius; by
            default, None.
        preprocessed : boolean, optional
            Whether filtered EDA data is being inputted; by default, True.
        peaks_ix : array_like, optional
            An optional array containing locations of SCR peaks; by default,
            None. If provided, an 'N SCRs' metric is included in the output.
        seg_size : int
            The segment size in seconds; by default, 60.
        rolling_window : int, optional
            The size, in seconds, of the sliding window across which to
            compute the EDA SQA metrics; by default, None.
        rolling_step : int, optional
            The step size, in seconds, of the sliding windows; by default, 15.
        show_progress : bool, optional
            Whether to show a progress bar; by default, True.

        Returns
        -------
        metrics : pd.DataFrame
            A DataFrame containing EDA quality assessment metrics by segment
            or sliding window.

        References
        ----------
        Kleckner, I.R., Jones, R. M., Wilder-Smith, O., Wormwood, J.B.,
        Akcakaya, M., Quigley, K.S., ... & Goodwin, M.S. (2017). Simple,
        transparent, and flexible automated quality assessment procedures for
        ambulatory electrodermal activity data. IEEE Transactions on Biomedical
        Engineering, 65(7), 1460-1467.
        """

        fs = self.fs
        seg_name = 'Moving Window' if rolling_window else 'Segment'
        metrics = []

        has_scr = peaks_ix is not None
        if has_scr:
            peaks_ix = np.asarray(peaks_ix, dtype = int)

        # Rolling window approach
        if rolling_window is not None:
            step = int(rolling_step * fs)
            win_len = int(rolling_window * fs)

            for i, start in enumerate(
                    tqdm(range(0, len(signal) - win_len + 1, step),
                         desc = 'EDA QA', disable = not show_progress)):
                end = start + win_len
                segment = signal[start:end]
                seg_temp = temp[start:end] if temp is not None else None

                # Run EDA QA by sliding window
                valid_ix, invalid_ix, seg_metrics = self._edaqa(
                    segment, seg_temp, preprocessed)
                total_len = len(segment)
                row = {
                    seg_name: i + 1,
                    'N Valid': len(valid_ix),
                    '% Valid': round((len(valid_ix) / total_len) * 100, 2),
                    'N Invalid': len(invalid_ix),
                    '% Invalid': round((len(invalid_ix) / total_len) * 100, 2),
                    **seg_metrics
                }
                if has_scr:
                    n_scr = np.count_nonzero((peaks_ix >= start) & (peaks_ix < end))
                    row['N SCRs'] = int(n_scr)
                metrics.append(row)

        # Segmented approach
        else:
            seg_len = int(seg_size * fs)
            n_segments = len(signal) // seg_len

            for i in range(n_segments):
                start, end = i * seg_len, (i + 1) * seg_len
                segment = signal[start:end]
                seg_temp = temp[start:end] if temp is not None else None

                # Run EDA QA by segment
                valid_ix, invalid_ix, seg_metrics = self._edaqa(
                    segment, seg_temp, preprocessed)
                total_len = len(segment)
                row = {
                    seg_name: i + 1,
                    'N Valid': len(valid_ix),
                    '% Valid': round((len(valid_ix) / total_len) * 100, 2),
                    'N Invalid': len(invalid_ix),
                    '% Invalid': round((len(invalid_ix) / total_len) * 100, 2),
                    **seg_metrics
                }
                if has_scr:
                    n_scr = np.count_nonzero((peaks_ix >= start) & (peaks_ix < end))
                    row['N SCRs'] = int(n_scr)
                metrics.append(row)
        metrics = pd.DataFrame(metrics)
        return metrics

    def _edaqa(
        self,
        signal,
        temp: Optional[np.ndarray] = None,
        preprocessed: bool = True
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Evaluate the input signal against Kleckner et al.'s (2017)
        quality rules."""

        # Filter EDA signal with a FIR low pass filter
        if not preprocessed:
            from EDA import Filters as eda_filters
            try:
                signal = eda_filters.lowpass_fir(signal)
            except ValueError:
                pass

            # Filter temperature data with a moving average filter
            if temp is not None:
                window = int(2 * self.fs)
                b = np.ones(window) / window
                temp = np.convolve(temp, b, mode = 'same')

        sampling_interval = 1 / self.fs
        total_len = len(signal)

        # Rule 1
        out_of_range_mask = self._check_out_of_range(signal)

        # Rule 2
        excessive_slope_mask = self._check_excessive_slope(
            signal, sampling_interval)

        # Rule 3
        temp_out_of_range_mask = None
        if temp is not None:
            if len(signal) != len(temp):
                temp = self._equalize_temp(signal, temp)
            temp_out_of_range_mask = self._check_temp_out_of_range(temp)

        # Combine rule masks
        if temp_out_of_range_mask is not None:
            invalid_mask = (out_of_range_mask | excessive_slope_mask |
                            temp_out_of_range_mask)
        else:
            invalid_mask = out_of_range_mask | excessive_slope_mask

        # Rule 4
        invalid_data = self._set_neighbors_invalid(
            invalid_mask, sampling_interval)

        # Get indices of valid and invalid data points
        valid_ix = np.where(~invalid_data)[0]
        invalid_ix = np.where(invalid_data)[0]

        # Compute metrics
        quality_metrics = {
            'Out of Range': np.sum(out_of_range_mask),
            '% Out of Range': round((np.sum(out_of_range_mask) / total_len) * 100, 2),
            'Excessive Slope': np.sum(excessive_slope_mask),
            '% Excessive Slope': round((np.sum(excessive_slope_mask) / total_len) * 100, 2),
            'Temp Out of Range': (np.sum(temp_out_of_range_mask)
                                  if temp_out_of_range_mask is not None
                                  else np.nan),
            '% Temp Out of Range': (round((np.sum(temp_out_of_range_mask) / total_len) * 100, 2)
                                    if temp_out_of_range_mask is not None
                                    else np.nan),
        }
        return valid_ix, invalid_ix, quality_metrics

    def _check_out_of_range(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """Return a boolean mask where EDA values are below eda_min or
        above eda_max (Rule 1)."""
        return (signal < self.eda_min) | (signal > self.eda_max)

    def _check_excessive_slope(
        self,
        signal: np.ndarray,
        sampling_interval: float
    ) -> np.ndarray:
        """Return a boolean mask where the slope exceeds eda_max_slope
        (Rule 2)."""
        slopes = np.concatenate([[0], np.diff(signal) / sampling_interval])
        return np.abs(slopes) > self.eda_max_slope

    def _check_temp_out_of_range(
        self,
        temp: Optional[np.ndarray] = None
    ) -> Union[None, np.ndarray]:
        """Return a boolean mask where temperature values are below temp_min
        or above temp_max (Rule 3)."""
        if temp is None:
            return None
        return (temp < self.temp_min) | (temp > self.temp_max)

    def _set_neighbors_invalid(
        self,
        invalid_mask: np.ndarray,
        sampling_interval: float
    ) -> np.ndarray:
        """Spread invalid labels ± invalid_spread_dur seconds around detected
        invalid points (Rule 4)."""
        invalid_spread_length = int(
            self.invalid_spread_dur / sampling_interval)
        spread = np.zeros_like(invalid_mask, dtype = bool)
        for d, flag in enumerate(invalid_mask):
            if flag:
                start_idx = max(d - invalid_spread_length, 0)
                end_idx = min(d + invalid_spread_length + 1, len(invalid_mask))
                spread[start_idx:end_idx] = True
        return spread

    def plot_validity(
        self,
        metrics: pd.DataFrame,
        title: Optional[str] = None,
    ) -> go.Figure:
        fig = go.Figure(
            data = [
                go.Bar(
                    x = metrics['Segment'],
                    y = metrics['% Invalid'],
                    name = 'Invalid',
                    marker = dict(color = 'tomato'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y}% '
                                    'invalid<extra></extra>'),
                go.Bar(
                    x = metrics['Segment'],
                    y = metrics['% Valid'],
                    name = 'Valid',
                    marker = dict(
                        color = 'white',
                        pattern = dict(
                            shape = '/',
                            fgcolor = '#4aba74',
                            size = 5,
                            solidity = 0.2
                        )
                    ),
                    hovertemplate = '<b>Segment %{x}:</b> %{y}% '
                                    'valid<extra></extra>')
            ]
        )

        # If N SCRs exist, add markers above bars
        if 'N SCRs' in metrics.columns:
            y_top = metrics['% Invalid'] + metrics['% Valid']
            mask = metrics['N SCRs'] > 0

            fig.add_trace(
                go.Scatter(
                    x = metrics.loc[mask, 'Segment'],
                    y = (y_top + 3).loc[mask],
                    mode = 'text+markers',
                    text = ['✦'] * mask.sum(),  # star only where SCRs exist
                    textposition = 'middle center',
                    textfont = dict(color = '#f9c669'),
                    marker = dict(size = 1, color = '#f9c669',
                                  symbol = 'circle'),
                    showlegend = False,
                    hovertemplate = 'SCR(s) detected<extra></extra>',
                )
            )

        fig.update_layout(
            barmode = 'stack',
            font = dict(family = 'Poppins', color = 'black'),
            xaxis = dict(
                title = dict(
                    text = 'Segment',
                    font = dict(size = 16),
                    standoff = 5),
                tickfont = dict(size = 14)
            ),
            yaxis = dict(
                title = dict(
                    text = 'Proportion',
                    font = dict(size = 16),
                    standoff = 2),
                tickfont = dict(size = 14)
            ),
            legend = dict(font = dict(size = 14), orientation = 'h',
                          yanchor = 'bottom', y = 1.05,
                          xanchor = 'right', x = 1.0),
            template = 'simple_white',
            margin = dict(l = 30, r = 15, t = 60, b = 50)
        )
        if title is not None:
            fig.update_layout(
                title = title
            )
        return fig

    def plot_quality_metrics(
        self,
        metrics: pd.DataFrame,
        title: Optional[str] = None,
    ) -> go.Figure:
        traces = [
            go.Bar(
                x = metrics['Segment'],
                y = metrics['% Out of Range'],
                name = 'EDA Out of Range',
                marker = dict(color = '#7cabcc'),
                hovertemplate = '<b>Segment %{x}:</b> %{y}% '
                                'EDA out of range<extra></extra>'),
            go.Bar(
                x = metrics['Segment'],
                y = metrics['% Excessive Slope'],
                name = 'Excessive Slope',
                marker = dict(color = '#ed77aa'),
                hovertemplate = '<b>Segment %{x}:</b> %{y}% '
                                'excessive slope<extra></extra>'),
        ]
        if '% Temp Out of Range' in metrics.columns:
            traces.append(
                go.Bar(
                    x = metrics['Segment'],
                    y = metrics['% Temp Out of Range'],
                    name = 'Temp Out of Range',
                    marker = dict(color = '#b095c2'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y}% temp out of range<extra></extra>'
                )
            )

        # Append '% Valid' trace
        traces.append(
            go.Bar(
                x = metrics['Segment'],
                y = metrics['% Valid'],
                name = 'Valid',
                marker = dict(
                    color = 'rgba(0,0,0,0)',
                    pattern = dict(
                        shape = '/',
                        fgcolor = '#4aba74',
                        size = 5,
                        solidity = 0.2)),
                hovertemplate = '<b>Segment %{x}:</b> %{y}% valid<extra></extra>'
            )
        )
        fig = go.Figure(data = traces)
        fig.update_layout(
            barmode = 'stack',
            font = dict(family = 'Poppins', color = 'black'),
            xaxis = dict(
                title = dict(
                    text = 'Segment',
                    font = dict(size = 16),
                    standoff = 5),
                tickfont = dict(size = 14)
            ),
            yaxis = dict(
                title = dict(
                    text = 'Proportion',
                    font = dict(size = 16),
                    standoff = 2),
                tickfont = dict(size = 14)
            ),
            legend = dict(font = dict(size = 14), orientation = 'h',
                          yanchor = 'bottom', y = 1.05,
                          xanchor = 'right', x = 1.0),
            template = 'simple_white',
            margin = dict(l = 30, r = 15, t = 60, b = 50)
        )
        if title is not None:
            fig.update_layout(
                title = title
            )
        return fig

    def _equalize_temp(self, eda, temp):
        """Interpolate or truncate data in the temperature array to match the
        length of the EDA data array."""
        eda_ix = np.arange(len(eda))
        temp_ix = np.arange(len(temp))
        if len(temp) < len(eda):
            interp_func = interp1d(temp_ix, temp, kind = 'linear',
                                   fill_value = 'extrapolate')
            temp = interp_func(eda_ix)
        if len(temp) > len(eda):
            temp = temp[:len(eda)]
        return temp