from typing import Literal
from scipy.signal import butter, cheby1, ellip, filtfilt, find_peaks, \
    hilbert, iirnotch, lfilter, sosfiltfilt
from scipy.ndimage import uniform_filter1d
import pandas as pd
import numpy as np

# ============================== ECG Filters =================================
class Filters:
    """
    A class for filtering raw electrocardiogram (ECG) signals.

    Parameters/Attributes
    ---------------------
    fs : int
        The sampling rate of the ECG signal.
    powerline_freq : int
        The powerline interference frequency. This value must be either
        50 or 60 (by default, 60). The 50 Hz power grid is prevalent in
        many European, Asian, and African countries.
    """
    def __init__(self,fs: int, powerline_freq: Literal[50, 60] = 60):
        """
        Initialize the Filters object.

        Parameters
        ----------
        fs : int
            The sampling rate of the ECG signal.
        powerline_freq : {50, 60}
            The powerline interference frequency. This value must be either
            50 or 60 (by default, 60). The 50 Hz power grid is prevalent in
            many European, Asian, and African countries.
        """
        self.fs = fs
        if powerline_freq not in [50, 60]:
            raise ValueError('The `powerline_freq` parameter must be set to '
                             '50 or 60 for the powerline interference '
                             'frequency.')
        else:
            self.pl_freq = powerline_freq

    def baseline_wander(
        self,
        signal: np.ndarray,
        cutoff: float = 0.05,
        order: int = 2
    ) -> np.ndarray:
        """
        Apply a high-pass filter to remove baseline wander from ECG data.

        Parameters
        ----------
        signal : array_like
            An array containing the noisy signal data.
        cutoff : float
            The cut-off frequency at which frequencies below this value
            in the signal are attenuated; by default, 0.05 Hz.
        order : int
            The filter order, i.e., the number of samples required to
            produce the desired filtered output; by default, 2.

        Returns
        -------
        filtered : array_like
            An array containing the filtered signal data.
        """
        nyquist = 0.5 * self.fs
        highcut = cutoff / nyquist
        b, a = butter(order, highcut, btype = 'high', analog = False)
        filtered = filtfilt(b, a, signal)
        return filtered

    def muscle_noise(
        self,
        signal: np.ndarray,
        lowcut: float = 30,
        highcut: float= 100,
        order: int = 2
    ) -> np.ndarray:
        """
        Apply a bandstop filter to remove muscle (EMG) noise from ECG data.

        Parameters
        ----------
        signal : array_like
            An array containing the noisy signal data.
        lowcut : float, optional
            The lower cutoff frequency of the bandstop filter;
            by default, 30.
        highcut : float, optional
            The upper cutoff frequency of the bandstop filter;
            by default, 100
        order : int, optional
            The filter order, i.e., the number of samples required to
            produce the desired filtered output; by default, 2.

        Returns
        -------
        filtered : array_like
            The filtered signal with muscle/EMG noise removed.
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(order, [low, high],
                     btype = 'bandstop',
                     analog = False,
                     output = 'sos')
        filtered = sosfiltfilt(sos, signal)
        return filtered

    def powerline_interference(
        self,
        signal: np.ndarray,
        q: float = 30
    ) -> np.ndarray:
        """
        Filter out powerline interference at a specified frequency.

        Parameters
        ----------
        signal : array_like
            An array containing the noisy signal data.
        q : float, optional
            The quality factor, i.e., how narrow or wide the stopband is
            for a notch filter (by default, 30). A higher quality factor
            indicates a narrower bandpass.

        Returns
        -------
        filtered : array_like
            An array containing the filtered signal data.
        """
        w0 = 2 * np.pi * self.pl_freq / self.fs
        b, a = iirnotch(w0, q)
        filtered = filtfilt(b, a, signal)
        return filtered

    def filter_signal(
        self,
        signal: np.ndarray,
        lowcut: float = 1,
        highcut: float = 15,
        rp: float= 0.15,
        rs: float = 80,
        order: int = 2
    ) -> np.ndarray:
        """
        Filter out artifact from ECG data due to powerline interference,
        baseline wander, movement, and muscle noise using an elliptic (Cauer)
        bandpass filter.

        Parameters
        ----------
        signal : array_like
            An array containing the ECG data to be filtered.
        lowcut : float, optional
            The lower cutoff frequency (in Hz) of the bandpass filter below
            which frequencies are attenuated; by default, 1 Hz.
        highcut : float, optional
            The upper cutoff frequency (in Hz) of the bandpass filter above
            which frequencies are attenuated; by default, 15 Hz.
        rp : float, optional
            The maximum ripple (in dB) allowed in the passband; by default,
            0.15 dB. Smaller values make the passband flatter (i.e.,
            allow less distortion of ECG frequencies).
        rs : float, optional
            The minimum ripple (in dB) allowed in the stopband; by default,
            80 dB. Larger values reject more noise.
        order : int, optional
            The filter order. Lower orders yield smoother transitions but less
            sharp cutoff; by default, 2.

        Returns
        -------
        filtered : array_like
            An array containing the bandpass-filtered ECG signal.

        Notes
        -----
        This is the default filter used in the PhysioView Dashboard.
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = ellip(order, rp, rs, [low, high], btype = 'band')
        filtered = filtfilt(b, a, signal)
        return filtered

# ======================= ECG Beat Detection Methods =========================
class BeatDetectors:
    """
    A class for detecting beats in electrocardiogram (ECG) signals using
    popular algorithms.

    Parameters/Attributes
    ---------------------
    fs : int
        The sampling rate of the ECG signal.
    preprocessed : bool, optional
        Whether the ECG signal is preprocessed or not; by default, True.

    Notes
    -----
    Both unfiltered and filtered ECG data may be passed as inputs to the
    beat detection functions. If unfiltered data is passed, the `preprocessed`
    parameter should be set to `False`, and the beat detection functions will
    pre-process the signal using the algorithm's original pre-processing
    procedures.
    """

    def __init__(self, fs: int, preprocessed: bool = True):
        """
        Initialize the BeatDetectors object.

        Parameters
        ----------
        fs : int
            The sampling rate of the ECG signal.
        preprocessed : bool, optional
            Whether the ECG signal is preprocessed or not; by default, True.
        """
        self.fs = fs
        if not isinstance(preprocessed, bool):
            raise ValueError(
                'The `preprocessed` attribute must be True or False.')
        else:
            self.preprocessed = preprocessed

    def engzee(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """
        Extract QRS complex locations from an ECG signal with the Engelse
         and Zeelenberg (1979) algorithm, modified by Lourenço et al. (2011).

        Parameters
        ----------
        signal : array_like
            An array containing the ECG signal.

        Returns
        -------
        ecg_beats : array_like
            An array containing the indices of detected R peaks.

        References
        ----------
        Engelse, W. A. H., & Zeelenberg, C. (1979). A single scan algorithm
        for QRS detection and feature extraction. IEEE Computers in
        Cardiology, 6, 37-42.

        Lourenço, A., Fred, A., & Ribeiro, B. (2012). Real time
        electrocardiogram segmentation for finger based ECG biometrics. In
        Proceedings of the 2012 Eighth International Conference on Intelligent
        Information Hiding and Multimedia Signal Processing (IIH-MSP)
        (pp. 21-24). IEEE.
        """

        if not self.preprocessed:
            # Pre-process data using built-in filters
            signal = Filters.filter_signal(signal, self.fs)
        else:
            pass

        # Differentiate the input signal
        diff = np.zeros(len(signal))
        for i in range(4, len(diff)):
            diff[i] = signal[i] - signal[i - 4]

        # Apply low-pass filter to the differentiated signal
        ci = [1, 4, 6, 4, 1]               # coefficients
        low_pass = lfilter(ci, 1, diff)

        # Zero out the first part of the signal to avoid edge effects
        low_pass[: int(0.2 * self.fs)] = 0

        # Define threshold parameters
        ms200 = int(0.2 * self.fs)
        ms1200 = int(1.2 * self.fs)
        ms160 = int(0.16 * self.fs)
        neg_threshold = int(0.01 * self.fs)

        # Initialize variables for threshold calculation
        M = 0
        M_list = []
        neg_m = []
        MM = []
        M_slope = np.linspace(1.0, 0.6, ms1200 - ms200)

        # Initialize lists for detected QRS complexes and R-peaks
        QRS = []
        ecg_beats = []

        # Initialize variables for peak detection
        counter = 0  # counter for tracking negative peaks
        thi_list = []  # list of indices where threshold is initially crossed
        thi = False  # if threshold is initially crossed
        thf_list = []  # list of indices where threshold is finally crossed
        thf = False  # if threshold is finallay crossed
        newM5 = False  # if new M5 value is calculated

        # Compensate for potential delay in peak detection
        engzee_fake_delay = 0

        # Loop through the low-pass filtered signal for peak detection
        for i in range(len(low_pass)):

            # Calculate threshold 'M' based on current QRS complex
            if i < 5 * self.fs:
                M = 0.6 * np.max(low_pass[: i + 1])
                MM.append(M)
                if len(MM) > 5:
                    MM.pop(0)
            elif QRS and i < QRS[-1] + ms200:
                newM5 = 0.6 * np.max(low_pass[QRS[-1]: i])
                if newM5 > 1.5 * MM[-1]:
                    newM5 = 1.1 * MM[-1]
            elif newM5 and QRS and i == QRS[-1] + ms200:
                MM.append(newM5)
                if len(MM) > 5:
                    MM.pop(0)
                M = np.mean(MM)
            elif QRS and i > QRS[-1] + ms200 and i < QRS[-1] + ms1200:
                M = np.mean(MM) * M_slope[i - (QRS[-1] + ms200)]
            elif QRS and i > QRS[-1] + ms1200:
                M = 0.6 * np.mean(MM)
            M_list.append(M)
            neg_m.append(-M)

            # Detect QRS complexes
            if not QRS and low_pass[i] > M:
                QRS.append(i)
                thi_list.append(i)
                thi = True
            elif QRS and i > QRS[-1] + ms200 and low_pass[i] > M:
                QRS.append(i)
                thi_list.append(i)
                thi = True

            # Check for negative threshold crossing within defined window
            if thi and i < thi_list[-1] + ms160:
                if low_pass[i] < -M and low_pass[i - 1] > -M:
                    thf = True
                if thf and low_pass[i] < -M:
                    thf_list.append(i)
                    counter += 1
                elif low_pass[i] > -M and thf:
                    counter = 0
                    thi = False
                    thf = False
            elif thi and i > thi_list[-1] + ms160:
                counter = 0
                thi = False
                thf = False

            # Check if the number of negative threshold crossings exceeds
            # the threshold
            if counter > neg_threshold:

                # Extract the unfiltered section around the detected peak
                unfiltered_section = signal[
                                     thi_list[-1] - int(0.01 * self.fs): i]

                # Detect R-peaks and append to the list
                ecg_beats.append(
                    engzee_fake_delay + np.argmax(unfiltered_section) +
                    thi_list[-1] - int(0.01 * self.fs))
                counter = 0
                thi = False
                thf = False

        # Remove the first detection as it requires QRS complex amplitude
        # for the threshold
        ecg_beats.pop(0)

        # Convert list of beats to numpy array and return
        ecg_beats = np.array(ecg_beats, dtype = 'int')

        ecg_beats = self._remove_dupes(ecg_beats)
        return ecg_beats

    def manikandan(
        self,
        signal: np.ndarray,
        adaptive_threshold: bool = True,
        window: float = 0.44
    ) -> np.ndarray:
        """
        Extract R peak locations from an ECG signal with the Manikandan
        and Soman (2012) algorithm.

        Parameters
        ----------
        signal : array_like
            An array containing the ECG signal.
        adaptive_threshold : boolean, optional
            Whether to refine beat detection using adaptive thresholding;
            by default, True.
        window : float, optional
            The size (in seconds) of the sliding search window for the
            adaptive threshold; by default, 0.44 seconds (440 milliseconds).

        Returns
        -------
        ecg_beats : array_like
            An array containing the indices of detected R peaks.

        References
        ----------
        Manikandan, R., & Soman, K. P. (2012). A novel method for detecting
        R-peaks in electrocardiogram (ECG) signal. Biomedical Signal Processing
        and Control, 7(2), 118-128.
        """
        def _adapt_thresh(signal, beats_ix, fs, window, step = 0.1):
            """Get beat indices with amplitudes that pass a minimum reference 
            threshold."""
            
            window_len = int(fs * window)
            window_step = int(fs * step)
            signal_beats = pd.DataFrame({'signal': signal})
            signal_beats.loc[beats_ix, 'beat'] = 1
            
            for n in range(0, len(signal_beats), window_step):
                w = signal_beats[n:(n + window_len)]
                if not w.beat.sum() >= 2:
                    pass
                else:
                    s = w['signal']
                    beats = w[w.beat == 1].index.values
                    if len(beats) == 2:
                        thresh = (s[beats].min() + s[beats].max()) * 0.5
                    if len(beats) > 2:
                        thresh = (s[beats].median() + s[beats].max()) * 0.5
                    reject = s[beats][s[beats] < thresh].index
                    signal_beats.loc[reject, 'beat'] = np.nan
            passing_beats = signal_beats[signal_beats.beat == 1].index.values
            return passing_beats

        if not self.preprocessed:
            signal = self._cheby1_filter(signal)
        else:
            pass
        signal = np.array(signal)

        # Differentiate the ECG signal
        dn = (np.append(signal[1:], 0) - signal)

        # Normalize the differentiated signal
        dtn = dn / (np.max(abs(dn)))

        # Compute their absolute values
        an = abs(dtn)

        # Compute their energy values
        en = an ** 2

        # Compute the Shannon entropy and energy values
        sen = -abs(dtn) * np.log10(abs(dtn))   # entropy
        sn = -(dtn ** 2) * np.log10(dtn ** 2)  # energy

        # Set the window size according to the duration of a QRS complex
        # (usually 120–150 ms)
        window_len = int(0.15 * self.fs)

        # Apply a moving average filter to the energy values
        sn_f = np.insert(self._ma_cumulative_sum(sn, window_len),
                         0, [0] * (window_len - 1))

        # Apply the Hilbert transform to the filtered energy values
        zn = np.imag(hilbert(sn_f))

        # Apply a moving average filter again to remove low-frequency drift
        # 2.5 sec from Manikandan & Soman (900 samples)
        # 2.5 sec in 500 Hz == 1250 samples
        ma_len = int(self.fs * 2.5)
        zn_ma = np.insert(
            self._ma_cumulative_sum(zn, ma_len), 0, [0] * (ma_len - 1))

        # Compute the difference to get only the high-frequency components
        zn_ma_s = zn - zn_ma

        # Look for zero-crossings: https://stackoverflow.com/a/28766902/6205282
        idx = np.argwhere(np.diff(np.sign(zn_ma_s)) > 0).flatten().tolist()

        # Prepare a container for windows
        idx_search = []
        ecg_beats = np.empty(0, dtype = int)
        search_window_half = round(self.fs * .12)
        for n in idx:
            lows = np.arange(n - search_window_half, n)
            highs = np.arange(n + 1, (n + search_window_half + 1))
            if highs[-1] >= len(signal):
                highs = np.delete(
                    highs, np.arange(
                        np.where(highs == len(signal))[0], len(highs)))
            ecg_window = np.concatenate((lows, [n], highs))
            idx_search.append(ecg_window)
            ecg_window_wave = signal[ecg_window]
            peak_loc = ecg_window[
                np.where(ecg_window_wave == np.max(ecg_window_wave))[0]]
            if peak_loc.item() > 0:
                ecg_beats = np.append(ecg_beats, peak_loc)

        ecg_beats = self._remove_dupes(ecg_beats)
        
        if adaptive_threshold is True:
            passing_beats = _adapt_thresh(signal, ecg_beats, self.fs, window)
            return passing_beats
        else:
            return ecg_beats

    def nabian(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """
        Extract R peak locations from an ECG signal with the Nabian
        et al. (2018) algorithm.

        Parameters
        ----------
        signal : array_like
            An array containing the ECG signal.

        Returns
        -------
        ecg_beats : array_like
            An array containing the indices of detected R peaks.

        References
        ----------
        Nabian, M., et al. (2018). An open-source feature extraction tool for
        the analysis of peripheral physiological data. IEEE Journal of
        Translational Engineering in Health and Medicine, 6, 1-11.
        """

        if not self.preprocessed:
            signal = self._elliptic_bandpass_filter(signal)
        else:
            pass

        window_size = int(0.4 * self.fs)
        peaks = np.zeros(len(signal))
        for i in range(1 + window_size, len(signal) - window_size):
            ecg_window = signal[i - window_size: i + window_size]
            rpeak = np.argmax(ecg_window)
            if i == (i - window_size - 1 + rpeak):
                peaks[i] = 1
        ecg_beats = np.where(peaks == 1)[0]
        ecg_beats = self._remove_dupes(ecg_beats)
        return ecg_beats

    def pantompkins(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """
        Extract QRS complex locations from an ECG signal with the
        Pan & Tompkins (1985) algorithm.

        Parameters
        ----------
        signal : array_like
            An array containing the ECG signal.

        Returns
        -------
        ecg_beats : array_like
            An array containing the indices of detected beats.

        References
        ----------
        Pan, S. J., & Tompkins, W. J. (1985). A real-time QRS detection
        algorithm. IEEE Transactions on Biomedical Engineering, 32(3), 230-236.
        """

        if not self.preprocessed:
            signal = self._butter_bandpass_filter(signal, method = 'pan')
        else:
            pass

        # Compute and square the differentiated signal
        diff = np.diff(signal)
        squared = diff * diff

        # Integrate the signal over a moving window of 150 ms in size
        window_size = int(0.15 * self.fs)
        mwa = uniform_filter1d(
            squared, window_size, origin = (window_size - 1) // 2)
        head_size = min(window_size - 1, len(squared))
        mwa[:head_size] = np.cumsum(signal[:head_size]) / np.linspace(
            1, head_size, head_size)
        mwa[: int(0.2 * self.fs)] = 0

        # Set the minimum distance criteria for peak validation
        min_peak_dist = int(0.3 * self.fs)
        min_missed_dist = int(0.25 * self.fs)

        # Initialize an array to store the beats
        ecg_beats = []

        # Preset running estimates of the signal and noise peaks
        SPKI = 0.0
        NPKI = 0.0

        last_peak = 0
        last_index = -1

        peaks, _ = find_peaks(mwa, plateau_size = (1, 1))
        for i, peak in enumerate(peaks):
            peak_value = mwa[peak]

            # Update the first threshold value
            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)

            # Check the current peak amplitude against the first threshold
            # and whether its location is greater than the minimum peak
            # distance from the last detected peak
            if (peak_value > threshold_I1) and peak > (
                    last_peak + min_peak_dist):
                ecg_beats.append(peak)

                # "Missed" IBI threshold is based on the previous eight IBIs
                if len(ecg_beats) > 9:
                    IBI_avg = (ecg_beats[-2] - ecg_beats[-10]) // 8
                    IBI_missed = int(1.66 * IBI_avg)

                    if (peak - last_peak) > IBI_missed:
                        missed_peaks = peaks[last_index + 1: i]

                        # Check the missed IBIs against the minimum missed
                        # IBI distance
                        missed_peaks = missed_peaks[
                            (missed_peaks > last_peak + min_missed_dist) & (
                                        missed_peaks < peak - min_missed_dist)]

                        # Update the second threshold value
                        threshold_I2 = 0.5 * threshold_I1

                        # Check the peak amplitude at the missed peak location
                        # against the second threshold
                        missed_peaks = missed_peaks[
                            mwa[missed_peaks] > threshold_I2]
                        if len(missed_peaks) > 0:
                            ecg_beats[-1] = missed_peaks[
                                np.argmax(mwa[missed_peaks])]
                            ecg_beats.append(peak)

                last_peak = peak
                last_index = i

                SPKI = 0.125 * peak_value + 0.875 * SPKI

            else:
                NPKI = 0.125 * peak_value + 0.875 * NPKI

        ecg_beats = self._remove_dupes(ecg_beats)
        return ecg_beats

    def _ma_cumulative_sum(
        self,
        signal: np.ndarray,
        window_len: int
    ) -> np.ndarray:
        """Moving average filter using cumulative sums."""
        cumsum = np.cumsum(np.insert(signal, 0, 0))
        ma = (cumsum[window_len:] - cumsum[:-window_len]) / float(window_len)
        return ma

    def _ma_convolution(
        self,
        signal: np.ndarray,
        window_len: int
    ) -> np.ndarray:
        """Moving average filter using convolution."""
        padded_diff = np.pad(
            signal, (window_len // 2, window_len // 2), mode = 'constant')
        ma = np.convolve(
            padded_diff, np.ones(window_len) / window_len, mode = 'valid')
        return ma

    def _butter_bandpass_filter(
        self,
        signal: np.ndarray,
        method: str
    ) -> np.ndarray:
        """A Butterworth bandpass filter, used in the preprocessing procedure
        of the Pan & Tompkins (1985) and Hamilton & Tompkins (1986) QRS
        detection algorithms. All parameters, including the passband frequency
        range (`Wn`) and order (`N`) of the filter, are provided as default
        values according to the algorithms."""
        if method not in ['pan', 'hamilton']:
            raise ValueError('The `method` parameter must take either \'pan\' '
                             'or \'hamilton\'.')
        else:
            if method == 'pan':
                lowcut = 0.5
                highcut = 15
            if method == 'hamilton':
                # Based on https://github.com/berndporr/py-ecg-detectors/
                lowcut = 8
                highcut = 16

        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(N = 2, Wn = [low, high], btype = 'band')
        preprocessed = filtfilt(b, a, signal)
        return preprocessed

    def _elliptic_bandpass_filter(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """An elliptic bandpass filter, used in the preprocessing procedure
        of the Nabian et al. (2018) R peak detection algorithm. All parameters,
        including the lower and upper cutoff frequencies (`Wn`), passband
        ripple (`rp`), stopband attenuation (`rs`), and order (`N`) of the
        filter, are provided as default values according to the algorithm."""
        nyq = 0.5 * self.fs
        lowcut = 0.5
        highcut = 50
        low = lowcut / nyq
        high = highcut / nyq
        b, a = ellip(N = 2, rs = 0.5, rp = 40, Wn = [low, high], btype = 'band')
        preprocessed = filtfilt(b, a, signal)
        return preprocessed

    def _cheby1_filter(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """A Chebyshev Type I bandpass filter, used in the preprocessing
        procedure of the Manikandan and Soman (2012) beat detection algorithm.
        All parameters, including the lower and upper cutoff frequencies
        (`Wn`), passband ripple (`rp`), and order (`N`), of the filter are
        provided as default values according to the algorithm."""
        nyquist = 0.5 * self.fs
        lowcut = 6
        highcut = 18
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = cheby1(N = 4, rp = 1, Wn = [low, high], btype = 'bandpass')
        preprocessed = filtfilt(b, a, signal)
        return preprocessed

    def _remove_dupes(
        self,
        ecg_beats: np.ndarray
    ) -> np.ndarray:
        """Remove duplicate indices in an `ecg_beats` array."""
        ecg_beats = np.array(ecg_beats)
        unique_values, unique_ix = np.unique(ecg_beats, return_index = True)
        dupe_removed = ecg_beats[sorted(unique_ix)]
        return dupe_removed