import numpy as np
from scipy.signal import butter, cheby2, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d

# ============================== PPG Filters =================================
class Filters:
    """
    A class for filtering raw photoplethysmography (PPG) data.

    Parameters/Attributes
    ---------------------
    fs : int
        The sampling rate of the PPG signal.
    """
    def __init__(self, fs: int):
        """
        Initialize the Filters object.

        Parameters
        ----------
        fs : int
            The sampling rate of the PPG signal.
        """
        self.fs = fs

    def baseline_wander(
        self,
        signal: np.ndarray,
        cutoff: float = 0.5,
        order: int = 2
    ) -> np.ndarray:
        """
        Apply a high-pass filter to remove baseline wander from PPG data.

        Parameters
        ----------
        signal : array_like
            An array containing the input PPG signal to be filtered.
        cutoff : float
            The cut-off frequency at which frequencies below this value
            in the PPG signal are attenuated; by default, 0.5 Hz.
        order : int
            The filter order, i.e., the number of samples required to
            produce the desired filtered output; by default, 2.
        """
        nyquist = 0.5 * self.fs
        highcut = cutoff / nyquist
        b, a = butter(order, highcut, btype = 'high', analog = False)
        filtered = filtfilt(b, a, signal)
        return filtered

    def moving_average(
        self,
        signal: np.ndarray,
        window_len: int
    ) -> np.ndarray:
        """
        Smooth a PPG signal using a moving average filter.

        Parameters
        ----------
        signal : array_like
            An array containing the input PPG signal to be filtered.
        window_len : int
            The size of the moving average window, in seconds.

        Returns
        -------
        filtered : array_like
            An array containing the filtered PPG signal.
        """
        kernel = np.ones(window_len) / window_len
        filtered = np.convolve(signal, kernel, mode = 'same')
        return filtered

    def filter_signal(
        self,
        signal: np.ndarray,
        lowcut: float = 0.5,
        highcut: float = 10,
        order: int = 4,
        window_len: float = 0.5
    ) -> np.ndarray:
        """
        Filter out baseline drift, motion artifact, and powerline
        interference from PPG data and smooth out the signal for further
        processing.

        Parameters
        ----------
        signal : array_like
            An array containing the input PPG signal to be filtered.
        lowcut : int, float
            The cut-off frequency at which frequencies below this value in the
            signal are attenuated; by default, 0.5 Hz.
        highcut : int, float
            The cut-off frequency at which frequencies above this value in the
            signal are attenuated; by default, 10 Hz.
        order : int
            The filter order, i.e., the number of samples required to
            produce the desired filtered output; by default, 4.
        window_len : int
            The size of the moving average window, in seconds.

        Returns
        -------
        filtered : array_like
            An array containing the filtered PPG data.

        References
        ----------
        Liang, Y., Elgendi, M., Chen, Z., et al. (2018). An optimal filter for
        short photoplethysmogram signals. Scientific Data, 5, 180076.

        Notes
        -----
        This function uses a 4th-order Chebyshev Type II filter according to
        results of the study by Liang et al. (2018) and a moving average
        filter using convolution. This is the default filter used in the
        Heartview Dashboard.
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = cheby2(order, 20, Wn = [low, high], btype = 'bandpass')
        filtered = filtfilt(b, a, signal)
        filtered_smoothed = self.moving_average(filtered, int(self.fs * window_len))
        return filtered_smoothed

# ======================= PPG Beat Detection Methods =========================
class BeatDetectors:
    """
    A class for detecting beats in photoplethysmography (PPG) signals using
    popular algorithms.

    Parameters/Attributes
    ---------------------
    fs : int
        The sampling rate of the PPG signal.
    preprocessed : bool, optional
        Whether the input PPG data is pre-processed or not; by default,
        `True`. If `False`, the beat detection methods will pre-process the
        input data according to the original algorithms.

    Notes
    -----
    Both unfiltered and filtered PPG data may be passed as inputs to the
    beat detection functions. If unfiltered data is passed, the `preprocessed`
    parameter should be set to `False`, and the beat detection functions will
    pre-process the signal using the algorithm's original pre-processing
    procedures.
    """

    def __init__(self, fs: int, preprocessed: bool = True):
        """
        Initialize a BeatDetectors object.

        Parameters
        ----------
        fs : int
            The sampling rate of the PPG signal.
        preprocessed : bool, optional
            Whether the PPG signal is preprocessed or not; by default, `True`.
            If `False`, the beat detection methods will pre-process the
            input data according to the original algorithms.
        """
        self.fs = fs
        if not isinstance(preprocessed, bool):
            raise ValueError(
                'The `preprocessed` attribute must be True or False.')
        else:
            self.preprocessed = preprocessed

    def adaptive_threshold(
        self,
        signal: np.ndarray,
        ma_perc: float = 20
    ) -> np.ndarray:
        """
        Extract beat locations from PPG data with the adaptive thresholding
        algorithm by van Gent al. (2018).

        Parameters
        ----------
        signal : array_like
            An array containing the PPG signal.
        ma_perc : float, optional
            The percentage with which to raise the moving average, used for
            fitting detection solutions to the data; by default, 20.

        Returns
        -------
        ppg_beats : array_like
            An array containing indices of PPG pulse onsets.

        Notes
        -----
        The original source code can be found in the HeartPy package at:
        https://github.com/paulvangentcom/heartrate_analysis_python.

        References
        ----------
        Van Gent, P., Farah, H., Van Nes, N., & Van Arem, B. (2019). HeartPy:
        A novel heart rate algorithm for the analysis of noisy signals.
        Transportation Research Part F: Traffic Psychology and Behaviour,
        66, 368-378.
        """
        if not self.preprocessed:
            filt = Filters(self.fs)
            signal = filt.filter_signal(signal)
        else:
            pass

        ma = self._moving_average(signal)
        rmean = np.array(ma)
        mn = np.mean(rmean / 100) * ma_perc
        ma = rmean + mn

        peaksx = np.where((signal > ma))[0]
        peaksy = signal[peaksx]
        peakedges = np.concatenate((np.array([0]),
                                    (np.where(np.diff(peaksx) > 1)[0]),
                                    np.array([len(peaksx)])))
        ppg_beats = []
        for i in range(0, len(peakedges) - 1):
            try:
                y_values = peaksy[peakedges[i]:peakedges[i + 1]].tolist()
                ppg_beats.append(
                    peaksx[peakedges[i] + y_values.index(max(y_values))])
            except:
                pass
        ppg_beats = np.asarray(ppg_beats).astype(int)
        return ppg_beats

    def erma(
        self,
        signal: np.ndarray,
        W1: float = 0.111,
        W2: float = 0.667,
        offset: float = 0.02,
        refractory: float = 0.3
    ) -> np.ndarray:
        """
        Extract beat locations from PPG data based on the Elgendi et al.
        (2013) algorithm using event-related moving averages and dynamic
        thresholding.

        Parameters
        ----------
        signal : array_like
            An array containing the PPG signal.
        W1 : float, optional
            The window size for peak detection in seconds; by default 111 ms.
        W2 : float, optional
            The window size for beat detection in seconds; by default, 667 ms.
        offset : float, optional
            Offset duration adjustment; by default, 20 ms.
        refractory : float, optional
            Refractory period to avoid double detection of the same beat
            (i.e., the minimum delay between consecutive systolic peaks); by
            default, 300 ms.

        Returns
        -------
        ppg_beats : array_like
            An array containing indices of PPG pulse onsets.

        References
        ----------
        Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D.
        (2013). Systolic peak detection in acceleration photoplethysmograms
        measured from emergency responders in tropical conditions. PLoS ONE,
        8(10), e76585.
        """

        if not self.preprocessed:
            signal = self._bandpass_filter(signal)
        else:
            pass

        signal_copy = np.copy(signal)

        # Clipping and squaring
        squared = np.maximum(signal_copy, 0) ** 2

        # Calculate peak and beat detection thresholds
        ma_kernel_peak = int(np.rint(W1 * self.fs))
        ma_kernel_beat = int(np.rint(W2 * self.fs))
        filt = Filters(self.fs)
        ma_peak = filt.moving_average(squared, ma_kernel_peak)
        ma_beat = filt.moving_average(squared, ma_kernel_beat)

        # Calculate threshold value
        thr1 = ma_beat + offset * np.mean(squared)

        # Identify start and end of PPG waves
        waves = ma_peak > thr1
        beg_waves = np.where(np.logical_and(~waves[:-1], waves[1:]))[0]
        end_waves = np.where(np.logical_and(waves[:-1], ~waves[1:]))[0]
        end_waves = end_waves[end_waves > beg_waves[0]]

        # Set duration criteria for peaks
        min_len = int(np.rint(W1 * self.fs))
        min_delay = int(np.rint(refractory * self.fs))

        # Identify systolic peaks within waves
        ppg_beats = [0]
        for beg, end in zip(beg_waves, end_waves):
            len_wave = end - beg
            # If the PPG wave is wider than the minimum duration
            if len_wave >= min_len:
                data = signal_copy[beg:end]
                local_max, props = find_peaks(data, prominence = (None, None))
                if local_max.size > 0:
                    peak = beg + local_max[np.argmax(props['prominences'])]
                    # If the IBI is greater than the refractory period
                    if peak - ppg_beats[-1] > min_delay:
                        ppg_beats.append(peak)

        ppg_beats.pop(0)
        ppg_beats = np.array(ppg_beats, dtype = 'int')
        return ppg_beats

    def _bandpass_filter(
        self,
        signal: np.ndarray,
        lowcut: float = 0.5,
        highcut: float = 8,
        order: int = 2
    ) -> np.ndarray:
        """A second-order bandpass filter to remove baseline wander from
        a PPG signal in the pre-processing procedure of the Elgendi et al.
        (2013) beat detection algorithm. All filter parameters are set as
        default values according to algorithm."""
        nyquist_freq = 0.5 * self.fs
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(order, [low, high], btype = 'band', analog = False)
        filtered = filtfilt(b, a, signal)
        return filtered

    def _moving_average(
        self,
        signal: np.ndarray,
        window_len: float = 0.75
    ) -> np.ndarray:
        """Compute the moving average of a signal in the van Gent et al.
        (2019) beat detection algorithm. The window length is set by default
        to 0.75 according to the algorithm."""
        ma = uniform_filter1d(np.asarray(signal, dtype = 'float'),
                              size = int(window_len * self.fs))
        return ma