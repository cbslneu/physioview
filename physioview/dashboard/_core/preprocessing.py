from typing import Literal, Optional, Tuple, Union
import importlib
import pandas as pd
import numpy as np

class Preprocessor:
    """
    A class for preprocessing physiological data in the PhysioView Dashboard.

    Parameters/Attributes
    ---------------------
    dtype : 'BVP', 'ECG', 'EDA', 'PPG'
        The type of data to be preprocessed. Must be one of 'BVP', 'ECG',
        'EDA', or 'PPG'.
    fs : int
        The sampling rate of the data.
    filter_on : bool
        Whether the signal should be filtered. This corresponds to the
        state of the BooleanSwitch component with the ID 'toggle-filter'.
    peak_detector : str, optional
            The name of peak detection algorithm to use.
    event_times : pd.DataFrame
        A DataFrame containing the uploaded event onsets and offsets,
        if any. If provided, must contain the columns: "event" (str),
        "start" (str, datetime), and "end" (str, datetime).
    seg_size : int, optional
        The size of the windows, in seconds, into which data will be
        segmented. Preprocesses the entire data or event window if not
        provided.
    """
    DEFAULT_FILTER_PARAMS = {
        'ECG': {
            'engzee': {
                'filt_method': 'filter_signal',
                'lowcut': 1, 'highcut': 15,
                'filt_type': 'Elliptic bandpass filter'},
            'manikandan': {
                'filt_method': 'cheby1_filter',
                'lowcut': 6, 'highcut': 18, 'order': 4, 'rp': 1,
                'filt_type': 'Chebyshev Type I bandpass filter'},
            'nabian': {
                'filt_method': 'elliptic_bandpass_filter',
                'lowcut': 0.5, 'highcut': 50, 'order': 2, 'rp': 0.5,
                'rs': 40, 'filt_type': 'Elliptic bandpass filter'},
            'pantompkins': {
                'filt_method': 'butter_bandpass_filter',
                'lowcut': 0.5, 'highcut': 15, 'order': 2,
                'filt_type': 'Butterworth bandpass filter'}
        },
        'PPG': {
            'filt_method': 'filter_signal',
            'lowcut': 0.5, 'highcut': 10, 'order': 4, 'window_len': 0.5,
            'filt_type': 'Chebyshev Type II filter with moving average '
                         'smoothing'
        },
        'EDA': {
            'filt_method': 'filter_signal',
            'cutoff': 0.35, 'filter_length': 2057, 'window_type': 'hamming',
            'filt_type': 'FIR low-pass filter'
        },
    }
    def __init__(
        self,
        dtype: Literal['BVP', 'ECG', 'EDA', 'PPG'],
        fs: int,
        filter_on: bool,
        peak_detector: Optional[str] = None,
        event_times: Optional[pd.DataFrame] = None,
        seg_size: Optional[int] = None,
        filter_kwargs: Optional[dict] = None
    ):
        """
        Initialize the Preprocessor object.

        Parameters
        ----------
        dtype : 'BVP', 'ECG', 'EDA', 'PPG'
            The type of data to be preprocessed. Must be one of 'BVP', 'ECG',
            'EDA', or 'PPG'.
        fs : int
            The sampling rate of the data.
        filter_on : bool
            Whether the signal should be filtered. This corresponds to the
            state of the BooleanSwitch component with the ID 'toggle-filter'.
        peak_detector : str, optional
            The name of peak detection algorithm to use.
        event_times : pd.DataFrame
            A DataFrame containing the uploaded event onsets and offsets,
            if any. If provided, must contain the columns: "event" (str),
            "start" (str, datetime), and "end" (str, datetime).
        seg_size : int, optional
            The size of the windows, in seconds, into which data will be
            segmented. Preprocesses the entire data or event window if not
            provided.
        filter_kwargs : dict, optional
            A dictionary containing filter keyword arguments relevant to the
            selected peak detector.
        """
        dtype = 'PPG' if dtype == 'BVP' else dtype
        if dtype not in ['ECG', 'EDA', 'PPG']:
            raise ValueError(f"dtype must be one of 'ECG', 'EDA', 'PPG'.")

        self.dtype = dtype
        self.fs = fs
        self.filter_on = filter_on
        self.peak_detector = peak_detector or self._get_default_peak_detector()
        self.event_times = event_times
        self.seg_size = seg_size
        self.filter_kwargs = filter_kwargs or {}
        self.duration = None

        # Initialize storage for peak and artifact indices
        self.peaks_ix = None
        self.artifacts_ix = None
        self.peaks_by_event = {}
        self.artifacts_by_event = {}

    def preprocess_full(
        self,
        data: pd.DataFrame,
        resample_rate: Optional[int] = None,
        min_peak_amp: Optional[float] = None,
        artifact_method: Optional[Literal['cbd', 'hegarty', 'both']] = None,
        artifact_tol: Optional[float] = None,
        temp_data: Optional[Union[list, np.ndarray, pd.Series]] = None,
        eda_min: Optional[float] = None,
        eda_max: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the full physiological data recording. If `self.seg_size`
        is specified, data is preprocessed by segment.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the uploaded physiological data. If a
            "Timestamp" column exists, timestamps must be `datetime`
            string-type values.
        resample_rate : int, optional
            The target sampling rate (Hz) for EDA resampling. If not provided,
            high-rate data (>8 Hz) is automatically downsampled to 8 Hz for
            computational efficiency, and low-rate data (≤8 Hz) keeps its
            original rate.
        min_peak_amp : float, optional
            The minimum amplitude, in microsiemens (µS), for an SCR peak to be
            considered valid. Used in the 'threshold' methodgy212 for EDA data;
            defaults to 0.05 µS if not provided.
        artifact_method : Literal['cbd', 'hegarty', 'both'], optional
            The user-selected method for identifying artifactual beats.
            Required for ECG and PPG data. This must be 'hegarty', 'cbd',
            or 'both'.
        artifact_tol : float, optional
            A configurable hyperparameter used to fine-tune the stringency of
            the criterion beat difference test. Used in the 'cbd' method for
            ECG and PPG artifact identification.
        temp_data : array_like
            An array containing temperature data, if any. Used in EDA
            quality assessment.
        eda_min : float, optional
            The minimum acceptable value for EDA data in microsiemens.
            Required for EDA data quality assessment.
        eda_max : float, optional
            The maximum acceptable value for EDA data in microsiemens.
            Required for EDA data quality assessment.

        Returns
        -------
        preprocessed : pd.DataFrame
            A DataFrame containing the preprocessed physiological data.
        sqa_metrics : pd.DataFrame
            A DataFrame containing the data quality summary.
        """
        preprocessed = data.copy()
        signal_col = self.dtype

        # Compute the signal duration in seconds
        self.duration = len(preprocessed) / self.fs

        # Filter signal
        if self.filter_on:
            signal_col = 'Filtered'
            preprocessed['Filtered'] = self._apply_filter(
                preprocessed[self.dtype])

        # Resample and decompose EDA signal before event segmentation
        if self.dtype == 'EDA':
            preprocessed = self._resample_eda(
                preprocessed, temp_data, resample_rate)
            preprocessed = self._decompose_eda(preprocessed)
            signal_col = 'Phasic'

        # Segment the data
        data_segments = self._segment_data(preprocessed)

        # Process each segment
        for seg_label, seg_data in data_segments.groupby('Segment'):

            # Detect peaks in the window
            seg_peaks, peak_label = self._detect_peaks(
                seg_data[signal_col],
                min_peak_amp = min_peak_amp if self.dtype == 'EDA' \
                    else None)

            # Add peak occurrence labels to full data
            peak_indices = seg_data.index[seg_peaks]
            preprocessed.loc[peak_indices, peak_label] = 1

        # Compute SQA metrics
        preprocessed, sqa_metrics = self._assess_signal_quality(
            preprocessed, artifact_method, artifact_tol, temp_data,
            eda_min, eda_max)

        return preprocessed, sqa_metrics

    def preprocess_event(
        self,
        data: pd.DataFrame,
        resample_rate: Optional[int] = None,
        min_peak_amp: Optional[float] = None,
        artifact_method: Optional[Literal['cbd', 'hegarty', 'both']] = None,
        artifact_tol: Optional[float] = None,
        temp_data: Optional[Union[list, np.ndarray, pd.Series]] = None,
        eda_min: Optional[float] = None,
        eda_max: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, dict, dict]:
        """
        Preprocess physiological data separately for each event window.

        This method filters the signal to each event window defined in
        `self.event_data` and then runs the preprocessing workflow to each
        event independently. If `self.seg_size` is specified, events are
        preprocessed by segment.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the uploaded physiological data. Must
            contain a "Timestamp" column with `datetime` string values.
        resample_rate : int, optional
            The target sampling rate (Hz) for EDA resampling. If not provided,
            high-rate data (>8 Hz) is automatically downsampled to 8 Hz for
            computational efficiency, and low-rate data (≤8 Hz) keeps its
            original rate.
        min_peak_amp : float, optional
            The minimum amplitude, in microsiemens (µS), for an SCR peak to be
            considered valid. Used in the 'threshold' method for EDA data;
            defaults to 0.05 µS if not provided.
        artifact_method : Literal['cbd', 'hegarty', 'both'], optional
            The user-selected method for identifying artifactual beats.
            Required for ECG and PPG data. This must be 'hegarty', 'cbd',
            or 'both'.
        artifact_tol : float, optional
            A configurable hyperparameter used to fine-tune the stringency of
            the criterion beat difference test. Used in the 'cbd' method for
            ECG and PPG artifact identification.
        temp_data : array_like
            An array containing temperature data, if any. Used in EDA
            quality assessment.
        eda_min : float, optional
            The minimum acceptable value for EDA data in microsiemens.
            Required for EDA data quality assessment.
        eda_max : float, optional
            The maximum acceptable value for EDA data in microsiemens.
            Required for EDA data quality assessment.

        Returns
        -------
        preprocessed : pd.DataFrame
            A DataFrame containing the preprocessed physiological data with
            results labeled by event in an "Event" column.
        preprocessed_by_event : dict
            A dictionary mapping events to their preprocessed data.
        sqa_metrics_by_event : dict
            A dictionary mapping events to their data quality summaries.
        """
        if self.event_times is None:
            raise ValueError('No event data provided.')
        if 'Timestamp' not in data.columns:
            raise ValueError("'Timestamp' column not found in data.")

        preprocessed = data.copy()
        signal_col = self.dtype
        preprocessed_by_event = {}
        sqa_metrics_by_event = {}

        # Compute the signal duration in seconds
        self.duration = len(preprocessed) / self.fs

        # Filter signal
        if self.filter_on:
            signal_col = 'Filtered'
            preprocessed['Filtered'] = self._apply_filter(
                preprocessed[self.dtype])

        # Resample and decompose EDA signal before event segmentation
        if self.dtype == 'EDA':
            preprocessed = self._resample_eda(
                preprocessed, temp_data, resample_rate)
            preprocessed = self._decompose_eda(preprocessed)
            signal_col = 'Phasic'

        # Insert event labels
        preprocessed['Timestamp'] = pd.to_datetime(preprocessed['Timestamp'])
        preprocessed.insert(1, 'Event', None)
        event_times = self.event_times.copy()
        for _, event in event_times.iterrows():
            event_label = event['event']
            event_start = event['start']
            event_end = event['end']
            preprocessed.loc[preprocessed['Timestamp'].between(
                event_start, event_end), 'Event'] = event_label

        # Preprocess each event
        for event_label, event_data in preprocessed.groupby('Event'):
            event_label: str
            event_data: pd.DataFrame

            # Preprocess the entire event
            if self.seg_size is None:

                # Detect all peaks in the event
                event_peaks, peak_label = self._detect_peaks(
                    event_data[signal_col], min_peak_amp = min_peak_amp \
                        if self.dtype == 'EDA' else None)

                # Add peak occurrence labels to full data
                peak_indices = event_data.index[event_peaks]
                event_data.loc[peak_indices, peak_label] = 1
                preprocessed.loc[peak_indices, peak_label] = 1

            # Preprocess each segment of the event
            else:

                # Segment each event
                event_segments = self._segment_data(event_data)

                # Process each segment within the event
                for seg_label, seg_data in event_segments.groupby('Segment'):

                    # Detect peaks in the window
                    seg_peaks, peak_label = self._detect_peaks(
                        seg_data[signal_col],
                        min_peak_amp = min_peak_amp if self.dtype == 'EDA' \
                            else None)

                    # Add peak occurrence labels to full data
                    peak_indices = seg_data.index[seg_peaks]
                    event_data.loc[peak_indices, peak_label] = 1
                    preprocessed.loc[peak_indices, peak_label] = 1

            # Compute and store SQA metrics for this event
            event_temp_data = event_data['Temp'].values if 'Temp' in event_data.columns else None
            event_data_updated, sqa_metrics = self._assess_signal_quality(
                event_data, artifact_method, artifact_tol, event_temp_data,
                eda_min, eda_max, event_label)
            preprocessed_by_event[event_label] = event_data_updated
            sqa_metrics_by_event[event_label] = sqa_metrics

            # Add SQA flags to full data
            if self.dtype == 'EDA':
                sqa_cols = [
                    'Invalid', 'Artifact', 'Out of Range', 'Excessive Slope']
            else:
                sqa_cols = ['Artifact']
            for col in sqa_cols:
                preprocessed.loc[event_data_updated.index, col] = \
                    event_data_updated[col]

        return preprocessed, preprocessed_by_event, sqa_metrics_by_event


    # ============================ Helper Methods ============================
    def _apply_filter(
        self,
        signal: Union[list, np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Apply a filter to the input signal based on dtype config."""
        filter_classes = {
            'ECG': ('physioview.pipeline.ECG', 'Filters'),
            'PPG': ('physioview.pipeline.PPG', 'Filters'),
            'EDA': ('physioview.pipeline.EDA', 'Filters')}

        # Get the default filter parameters
        if self.dtype == 'ECG':
            params = self.DEFAULT_FILTER_PARAMS['ECG'][self.peak_detector]
        else:
            params = self.DEFAULT_FILTER_PARAMS[self.dtype]
        filter_kwargs = {k: v for k, v in params.items()
                         if k not in ('filt_type', 'filt_method')}

        # Update with any user-customized filter parameters
        filter_kwargs.update(self.filter_kwargs)

        # Get the filter method from the signal type's Filters class
        filt_method = params.get('filt_method', 'filter_signal')

        # Apply the filter with its specific filter parameters
        module_name, class_name = filter_classes[self.dtype]
        module = importlib.import_module(module_name)
        FilterClass = getattr(module, class_name)
        method = getattr(FilterClass(self.fs), filt_method)
        return method(signal, **filter_kwargs)

    def _get_default_peak_detector(self) -> dict:
        """Get the default peak detector for the signal type."""
        peak_detector = {
            'ECG': 'manikandan',
            'PPG': 'adaptive_threshold',
            'EDA': 'threshold'
        }
        return peak_detector.get(self.dtype)

    def _check_unix(
        self,
        timestamps: Union[list, np.ndarray, pd.Series]
    ) -> Union[str, None]:
        """Check whether an array of timestamps contains Unix timestamps in
        s, ms, or µs."""
        try:
            ts = pd.to_numeric(timestamps, errors = 'coerce').dropna()
        except Exception:
            return None
        if ts.empty:
            return None
        median_val = ts.median()
        if 1e8 < median_val < 2e9:
            return 's'
        elif 1e11 < median_val < 2e13:
            return 'ms'
        elif 1e14 < median_val < 2e16:
            return 'us'
        else:
            return None

    def _segment_data(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add segment labels based on a given segment size.

        Parameters
        ----------
        data : pd.DataFrame
            The physiological data to segment.

        Returns
        -------
        data : pd.DataFrame
            The existing data with an added 'Segment' column containing
            segment labels.
        """
        window_samples = self.seg_size * self.fs
        segment_labels = np.arange(len(data)) // window_samples + 1
        data.insert(0, 'Segment', segment_labels)
        return data

    def _detect_peaks(
        self,
        signal: Union[list, np.ndarray, pd.Series],
        min_peak_amp: Optional[float] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Detect peaks in the input signal based on dtype config.

        Parameters
        ----------
        signal : array_like
            An array containing the physiological signal to detect peaks in.
        min_peak_amp : float, optional
            The minimum amplitude, in microsiemens (µS), for an SCR peak to be
            considered valid. Used in the 'threshold' method; defaults to
            0.05 µS if not provided.

        Returns
        -------
        peaks : np.ndarray or dict
            A NumPy array of peak indices.
        peak_name : str
            The label for the peak type: 'Beat' for ECG/PPG; 'SCR' for EDA.
        """
        peak_detectors = {
            'ECG': ('physioview.pipeline.ECG', 'BeatDetectors'),
            'PPG': ('physioview.pipeline.PPG', 'BeatDetectors'),
            'EDA': ('physioview.pipeline.EDA', 'detect_scr_peaks'),
        }
        module_name, attr_name = peak_detectors[self.dtype]
        module = importlib.import_module(module_name)
        detector_attr = getattr(module, attr_name)

        if self.dtype in ['ECG', 'PPG', 'BVP']:
            DetectorClass = detector_attr(self.fs, self.filter_on)

            # Get the method by name
            # - ECG: 'manikandan' (default), 'pantompkins', 'engzee', 'nabian'
            # - PPG: 'adaptive_threshold' (default), 'erma'
            # - EDA: 'threshold' (default), 'nabian'
            beat_detection_method = getattr(DetectorClass, self.peak_detector)
            peaks = beat_detection_method(signal)
            peak_name = 'Beat'

        elif self.dtype == 'EDA':
            peaks = detector_attr(
                signal, method = self.peak_detector, fs = self.fs,
                min_peak_amp = min_peak_amp)
            peak_name = 'SCR'

        else:
            raise ValueError(
                f"Cannot detect peaks in signal of dtype '{self.dtype}'.")

        peaks = np.asarray(peaks)
        if peaks.size == 0:
            peaks = np.array([], dtype = int)
        else:
            peaks = peaks.astype(int)
        return peaks, peak_name

    def _resample_eda(
        self,
        data: pd.DataFrame,
        temp_data: Optional[Union[list, np.ndarray, pd.Series]] = None,
        resample_rate: int = 8
    ) -> pd.DataFrame:
        """Resample EDA data to a target sampling rate and update self.fs
        to the resampled rate for use in subsequent preprocessing."""
        signal_cols = ['EDA']
        if self.filter_on:
            signal_cols.append('Filtered')

        from physioview.pipeline.EDA import resample

        # Create resampled data
        data_resampled = pd.DataFrame()
        target_fs = resample_rate if resample_rate is not None \
            else (8 if self.fs > 8 else self.fs)
        if target_fs != self.fs:
            for col in signal_cols:
                signal_rs = resample(data[col], self.fs, target_fs)
                data_resampled[col] = signal_rs
            if temp_data is not None:
                temp_rs = resample(temp_data, self.fs, target_fs)

            # Update the original sampling rate to the resample rate
            self.fs = target_fs

        else:
            data_resampled = data[signal_cols].copy()
            temp_rs = temp_data

        # Build resampled timestamps/sample indices
        n_samples = len(data_resampled)
        if 'Timestamp' in data.columns:
            ts_col = 'Timestamp'
            t0 = data['Timestamp'].iloc[0]
            step = pd.to_timedelta(1, unit = 's') / self.fs
            ts_rs = pd.date_range(
                start = t0, periods = n_samples, freq = step)
        else:
            ts_col = 'Sample'
            ts_rs = np.arange(n_samples)
        data_resampled.insert(0, ts_col, ts_rs)

        # Add resampled temperature signal
        if temp_rs is not None:
            data_resampled['Temp'] = temp_rs

        return data_resampled

    def _decompose_eda(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Decompose EDA signal into phasic and tonic components and add
        'Phasic', 'Tonic', and 'Decomposed' columns accordingly."""
        if self.dtype != 'EDA':
            return data

        from physioview.pipeline.EDA import decompose_eda
        phasic, tonic = decompose_eda(
            data[self.dtype].values, self.fs, show_progress = False)
        data['Phasic'] = phasic
        data['Tonic'] = tonic
        data['Decomposed'] = phasic + tonic
        return data

    def _assess_signal_quality(
        self,
        data: pd.DataFrame,
        artifact_method: Optional[Literal['cbd', 'hegarty', 'both']] = None,
        artifact_tol: Optional[float] = None,
        temp_data: Optional[Union[list, np.ndarray, pd.Series]] = None,
        eda_min: Optional[float] = None,
        eda_max: Optional[float] = None,
        event_label: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assess and flag artifactual beats or invalid data points in
        physiological data.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the physiological signal to assess
            for quality.
        artifact_method : Literal['cbd', 'hegarty', 'both'], optional
            The user-selected method for identifying artifactual beats.
            Required for ECG and PPG data. This must be 'hegarty', 'cbd',
            or 'both'.
        artifact_tol : float, optional
            A configurable hyperparameter used to fine-tune the stringency of
            the criterion beat difference test. Used in the 'cbd' method for
            ECG and PPG artifact identification.
        temp_data : array_like
            An array containing temperature data, if any. Used in EDA
            quality assessment.
        eda_min : float, optional
            The minimum acceptable value for EDA data in microsiemens.
            Required for EDA data quality assessment.
        eda_max : float, optional
            The maximum acceptable value for EDA data in microsiemens.
            Required for EDA data quality assessment.
        event_label : str, optional
            The name of the event window being assessed. Required for
            storing peak and artifact indices per event.

        Returns
        -------
        data : pd.DataFrame
            A DataFrame containing the existing data with columns
            corresponding to signal quality features (e.g., "Artifact",
            "Invalid", "Excessive Slope").
        sqa_metrics : pd.DataFrame
            A DataFrame containing the signal quality assessment summary.
        """
        ts_col = 'Timestamp' if 'Timestamp' in data.columns else None
        has_event_times = self.event_times is not None \
                          and not self.event_times.empty
        seg_size = (len(data) / self.fs) if \
            (not self.seg_size and has_event_times) else self.seg_size

        if self.dtype in ['ECG', 'PPG', 'BVP']:
            if not artifact_method or not artifact_tol:
                raise ValueError('Missing artifact_method or artifact_tol.')

            from physioview.pipeline.SQA import Cardio
            sqa_cardiac = Cardio(self.fs)
            peaks_ix = data[data['Beat'] == 1].index.values
            artifacts_ix = sqa_cardiac.identify_artifacts(
                peaks_ix, method = artifact_method, tol = artifact_tol)
            data.loc[artifacts_ix, 'Artifact'] = 1

            # Compute SQA metrics
            sqa_metrics = sqa_cardiac.compute_metrics(
                data, peaks_ix, artifacts_ix, ts_col = ts_col,
                seg_size = seg_size, show_progress = False)

        elif self.dtype == 'EDA':
            if not eda_min or not eda_max:
                raise ValueError('Missing eda_min or eda_max.')

            from physioview.pipeline.SQA import EDA
            sqa_eda = EDA(fs = self.fs, eda_min = eda_min, eda_max = eda_max)
            signal_col = 'Filtered' if self.filter_on else self.dtype
            eda_validity = sqa_eda.get_validity_metrics(
                signal = data[signal_col].values, temp = temp_data,
                preprocessed = self.filter_on)
            data.loc[eda_validity['Invalid'].notna().values, 'Invalid'] = 1

            # Compute SQA metrics
            eda_quality = sqa_eda.get_quality_metrics(data[signal_col].values)
            data[eda_quality.columns[-2:]] = eda_quality[
                eda_quality.columns[-2:]].values
            artifacts_ix = data[
                (data['Invalid'] == 1) & (data['SCR'] == 1)].index.values
            data.loc[artifacts_ix, 'Artifact'] = 1
            peaks_ix = data[data['SCR'] == 1].index.values

            sqa_metrics = sqa_eda.compute_metrics(
                data[signal_col].reset_index(drop = True), temp_data,
                self.filter_on, peaks_ix, seg_size = seg_size,
                show_progress = False)

        else:
            raise ValueError(
                f"Signal quality assessment not supported for dtype "
                f"'{self.dtype}'.")

        # Store peak and artifact indices
        if has_event_times and event_label:
            if not hasattr(self, 'peaks_by_event'):
                self.peaks_by_event = {}
            if not hasattr(self, 'artifacts_by_event'):
                self.artifacts_by_event = {}
            self.peaks_by_event[event_label] = peaks_ix
            self.artifacts_by_event[event_label] = artifacts_ix
        else:
            self.peaks_ix = peaks_ix
            self.artifacts_ix = artifacts_ix

        return data, sqa_metrics