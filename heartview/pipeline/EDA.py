import numpy as np
import pandas as pd
import cvxopt as cv
from typing import Literal, Optional, Union
from zipfile import ZipFile
from scipy.signal import butter, convolve, ellip, filtfilt, find_peaks, \
    firwin, resample_poly
from math import gcd
from scipy.fft import fft, ifft, fftfreq

# ============================== EDA Filters =================================
class Filters:
    """
    A class for filtering raw electrodermal activity (EDA) data.

    Parameters/Attributes
    ---------------------
    fs : int
        The sampling rate of the EDA signal.
    """

    def __init__(self, fs):
        """
        Initialize the Filters object.

        Parameters
        ----------
        fs : int
            The sampling rate of the PPG signal.
        """
        self.fs = fs

    def lowpass_butter(self, signal, cutoff = 2, order = 3):
        """
        Filter an EDA signal with a Butterworth low-pass filter.

        Parameters
        ----------
        signal : array_like
            An array containing the raw EDA signal.
        cutoff : int, float
            The cut-off frequency at which frequencies above this value in the
            EDA signal are attenuated; by default, 2 Hz.
        order : int
            The filter order, i.e., the number of samples required to produce
            the desired filtered output; by default, 3.

        Returns
        -------
        filtered : array_like
            An array containing the filtered EDA signal.
        """
        nyq = 0.5 * self.fs
        normalized_cutoff = cutoff / nyq
        b, a = butter(order, normalized_cutoff, btype = 'low', analog = False)
        filtered = filtfilt(b, a, signal)
        return filtered

    def lowpass_elliptic(
        self,
        signal: np.ndarray,
        cutoff: float = 1.0,
        order: int = 4,
        rp: float = 1,
        rs: float = 40
    ) -> np.ndarray:
        """
        Apply an elliptic low-pass filter to an EDA signal.

        Parameters
        ----------
        signal : array_like
            An array containing the raw EDA signal to be filtered.
        cutoff : float, optional
            The cutoff frequency in Hz; by default, 1 Hz.
        order : int, optional
            The filter order; by default, 4.
        rp : float, optional
            Maximum ripple (in dB) allowed in the passband; by default, 1 dB.
        rs : float, optional
            Minimum attenuation (in dB) required in the stopband;
            by default, 40 dB.

        Returns
        -------
        filtered : array_like
            An array containing the filtered EDA signal.
        """
        nyq = 0.5 * self.fs
        wn = cutoff / nyq
        b, a = ellip(order, rp, rs, wn, btype = 'low')
        filtered = filtfilt(b, a, signal)
        return filtered

    def lowpass_gaussian(self, signal, cutoff: float = 1.0):
        """
        Apply a frequency-domain Gaussian low-pass filter to an EDA signal.

        Parameters
        ----------
        signal : array_like
            An array containing the raw EDA signal.
        cutoff : float, optional
            The desired cutoff frequency in Hz; by default, 1.0 Hz.

        Returns
        -------
        filtered : array_like
            An array containing the low-pass filtered EDA signal.

        References
        ----------
        Nabian, M., et al. (2018). An open-source feature extraction tool for
        the analysis of peripheral physiological data. IEEE Journal of
        Translational Engineering in Health and Medicine, 6, 1-11.
        """

        n = len(signal)
        freqs = fftfreq(n, d = 1 / self.fs)

        # Compute the FFT of the EDA signal
        sig_fft = fft(signal)

        # Compute the Gaussian frequency response
        # Attenuate frequencies above cutoff symmetrically for +/- freqs
        gaussian_response = np.exp(-0.5 * (freqs / cutoff) ** 2)

        # Filter in the frequency domain
        sig_fft_filtered = sig_fft * gaussian_response

        # Transform to time domain
        filtered = np.real(ifft(sig_fft_filtered))
        return filtered

    def filter_signal(
        self,
        signal: np.ndarray,
        fs: int = 4,
        cutoff: float = 0.35,
        filter_length: int = 2057,
        window_type: Literal['hamming', 'hann', 'blackman'] = 'hamming'
    ) -> np.ndarray:
        """
        Apply a FIR low-pass filter to an EDA signal.

        Parameters
        ----------
        signal : array_like
            An array containing the raw EDA signal to be filtered.
        fs : int, optional
            The sampling rate of the signal in Hz; by default, 4 Hz.
        cutoff : float, optional
            The cut-off frequency above which frequencies are attenuated;
            by default, 0.35 Hz.
        filter_length : int, optional
            The length of the FIR filter (number of taps); by default, 2057.
            Larger values give smoother filtering but increase computation
            and delay.
        window_type : {'hamming', 'hann', 'blackman'}, optional
            The type of window function to use for FIR design;
            by default, 'hamming'.

        Returns
        -------
        filtered : array_like
            An array containing the filtered EDA signal.

        References
        ----------
        Kleckner, I.R., Jones, R.M., Wilder-Smith, O., Wormwood, J.B.,
        Akcakaya, M., Quigley, K.S., ... & Goodwin, M.S. (2017). Simple,
        transparent, and flexible automated quality assessment procedures for
        ambulatory electrodermal activity data. IEEE Transactions on
        Biomedical Engineering, 65(7), 1460-1467.

        Notes
        -----
        This is the default filter used in the PhysioView Dashboard.
        """

        # Normalize cutoff frequency to Nyquist
        cutoff_norm = cutoff / (fs / 2)

        # Design FIR filter
        fir_coeff = firwin(
            numtaps = filter_length,
            cutoff = cutoff_norm,
            window = window_type
        )

        # Apply filter to signal
        filtered = filtfilt(fir_coeff, [1.0], signal)
        return filtered

    def moving_average(self, signal, window_len):
        """
        Apply a moving average filter to an EDA signal.

        Parameters
        ----------
        signal : array_like
            An array containing the raw EDA signal.
        window_len : int or float
            The moving average window size, in seconds.

        Returns
        -------
        ma : array_like
            An array containing the moving average of the EDA signal.
        """
        samples = int(window_len * self.fs)
        kernel = np.ones(samples) / samples
        ma = np.convolve(signal, kernel, mode = 'same')
        epsilon = np.finfo(float).eps   # pad the beginning with epsilons
        ma = np.concatenate((ma, np.full(len(signal) - len(ma), epsilon)))
        return ma

# ======================== Other EDA Data Processing ========================
def detect_scr_peaks(
    phasic: Union[np.ndarray, pd.Series],
    smooth_size: int = 20,
    min_amp_thresh: float = 0.1,
    min_peak_amp: Optional[float] = None,
) -> np.ndarray:
    """
    Detect skin conductance response (SCR) peaks in a phasic EDA signal using
    the Nabian et al. (2018) approach.

    Parameters
    ----------
    phasic : array-like
        Phasic component of EDA signal.
    smooth_size : int, optional
        The size of Bartlett window for smoothing derivative; by default, 20.
    min_amp_thresh : float, optional
        The minimum SCR amplitude threshold, relative to the max detected
        amplitude; by default, 0.1 (i.e., 10%).
    min_peak_amp : float, optional
        The minimum amplitude for a SCR peak to be considered valid; by
        default, None.

    Returns
    -------
    peaks : np.ndarray
        An array containing indices of detected SCR peaks.

    References
    ----------
    Nabian, M., et al. (2018). An open-source feature extraction tool for
    the analysis of peripheral physiological data. IEEE Journal of
    Translational Engineering in Health and Medicine, 6, 1-11.
    """

    # Differentiate phasic signal
    diff = np.diff(phasic, prepend = phasic[0])

    # Smooth derivative with Bartlett kernel
    kernel = np.bartlett(smooth_size)
    kernel /= kernel.sum()
    diff_smoothed = convolve(diff, kernel, mode="same")

    # Find zero crossings
    def _get_zero_crossings(sig, direction = 'positive'):
        zc = np.where(np.diff(np.sign(sig)) != 0)[0]
        if direction == 'positive':
            return [i for i in zc if sig[i] < 0 and sig[i+1] >= 0]
        elif direction == 'negative':
            return [i for i in zc if sig[i] > 0 and sig[i+1] <= 0]
        else:
            return zc

    pos_crossings = _get_zero_crossings(diff_smoothed, 'positive')
    neg_crossings = _get_zero_crossings(diff_smoothed, 'negative')

    # Ensure onset before offset
    if len(neg_crossings) > 0 and len(pos_crossings) > 0:
        if neg_crossings[0] < pos_crossings[0]:
            neg_crossings = neg_crossings[1:]
    n_pairs = min(len(pos_crossings), len(neg_crossings))
    pos_crossings = pos_crossings[:n_pairs]
    neg_crossings = neg_crossings[:n_pairs]

    # Get peaks
    candidates = []
    for onset, offset in zip(pos_crossings, neg_crossings):
        window = phasic[onset:offset]
        if len(window) == 0:
            continue
        peak_idx = onset + np.argmax(window)
        amp = phasic[peak_idx] - phasic[onset]
        candidates.append((peak_idx, amp))

    # Apply sequential amplitude threshold
    if len(candidates) == 0:
        return np.array([])
    peaks, amps = [], []
    for idx, amp in candidates:
        if not amps:
            if (min_peak_amp is None) or (amp >= min_peak_amp):
                peaks.append(idx)
                amps.append(amp)
        else:
            use_rel = amp >= (min_amp_thresh * max(amps))
            use_abs = (min_peak_amp is None) or (amp >= min_peak_amp)
            if use_abs and use_rel:
                peaks.append(idx)
                amps.append(amp)

    return np.array(peaks)

def compute_tonic_scl(
    signal: np.ndarray,
    fs: int = 4,
    seg_size: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Compute the tonic skin conductance level (SCL) as the mean of the EDA
    signal over a given segment, while excluding intervals corresponding to
    skin conductance responses (SCRs).

    Parameters
    ----------
    signal: array_like
        An array containing the filtered EDA signal.
    fs : int, optional
        The sampling rate of the EDA signal; by default, 4 Hz.
    seg_size : int, optional
        The segment size (in seconds) of the EDA signal; by default, None.
        If given, the EDA signal is divided into non-overlapping segments of
        this length and a tonic SCL is computed for each segment. If None,
        the tonic SCL is computed once across the whole signal.

    Returns
    -------
    tonic_scl : float or np.ndarray
        The tonic skin conductance level (SCL).

    Notes
    -----
    SCRs are detected based on amplitude and temporal criteria, and the data
    from their onset through recovery are omitted from the calculation. If no
    SCRs are detected, the tonic SCL is equivalent to the mean of the entire
    signal.
    """
    def _scr_intervals(
        min_height: float = 0.05,
        min_rise_time: float = 1.0,
        min_recovery_time: float = 2.0
    ) -> list[tuple]:
        """
        Detect SCR intervals (start, end) from an EDA signal.

        Parameters
        ----------
        min_height : float, optional
            The minimum SCR amplitude (in µS) to be considered valid;
            by default, 0.05 uS.
        min_rise_time : float, optional
            Minimum time (in seconds) from SCR onset to peak;
            by default, 1 second.
        min_recovery_time : float, optional
            Minimum time (in seconds) from peak back to baseline;
            by default, 2 seconds.

        Returns
        -------
        scr_intervals : list of tuple
            A list of (start_index, end_index) intervals for each detected
            SCR.
        """

        # Differentiate signal to emphasize rises
        diff_sig = np.diff(signal, prepend = signal[0])

        # Detect candidate peaks in the EDA signal itself
        min_distance = int((min_rise_time + min_recovery_time) * fs)
        peaks, props = find_peaks(signal, height = min_height,
                                  distance = min_distance)

        # Get start and end indices of SCR intervals
        scr_intervals = []
        for peak in peaks:
            start = peak
            # where derivative last went negative before the peak
            while start > 0 and diff_sig[start] > 0:
                start -= 1
            end = peak
            # where derivative first goes negative after the peak
            while end < len(signal) - 1 and diff_sig[end] < 0:
                end += 1
            scr_intervals.append((start, end))
        return scr_intervals

    def _masked_mean(sig_segment: np.ndarray) -> float:
        """Compute mean excluding SCR intervals."""
        scr_intervals = _scr_intervals()
        mask = np.ones(len(sig_segment), dtype = bool)
        for start, end in scr_intervals:
            mask[start:end] = False
        return np.mean(sig_segment[mask]) if np.any(mask) else np.nan

    if seg_size is None:
        # One tonic SCL for the whole signal
        tonic_scl = _masked_mean(signal)
    else:
        # Segmented tonic SCL
        seg_len = int(seg_size * fs)
        n_segments = len(signal) // seg_len
        tonic_scl = []
        for i in range(n_segments):
            start, end = i * seg_len, (i + 1) * seg_len
            tonic_scl.append(_masked_mean(signal[start:end]))
        tonic_scl = np.array(tonic_scl)

    return tonic_scl

def decompose_eda(
    signal: np.ndarray,
    fs: int,
    show_progress: bool = True
) -> tuple:
    """
    Extract the phasic and tonic components of an electrodermal activity (EDA)
    signal using the convex optimization approach by Greco et al. (2015).
    This is an alias function for `cvxEDA()` in this module.

    Parameters
    ----------
    signal : array_like
        An array containing the EDA signal.
    fs : float
        The sampling rate of the EDA signal.

    Returns
    -------
    phasic : array_like
        The phasic component (fast-moving changes) of the EDA signal.
    tonic : array_like
        The tonic component (slow-moving changes) of the EDA signal.

    References
    ----------
    A Greco, G. Valenza, A. Lanata, E. P. Scilingo, & L. Citi. (2015). cvxEDA:
    A convex optimization approach to electrodermal activity processing. IEEE
    Transactions on Biomedical Engineering, 63(4): 797-804.
    """
    phasic, _, tonic, _, _, _, _ = _cvxEDA(
        signal, fs, options = {'show_progress': show_progress})
    return phasic, tonic

def resample(
    signal: np.ndarray,
    fs: int,
    new_fs: int
) -> np.ndarray:
    """
    Resample a signal to a new sampling frequency using polyphase filtering.

    Parameters
    ----------
    signal : array_like
        The input signal (1D array).
    fs : int
        The original sampling frequency of the signal.
    new_fs : int
        The desired sampling frequency.

    Returns
    -------
    rs : array_like
        The resampled signal.
    """
    # Ensure integer values
    fs = int(round(fs))
    new_fs = int(round(new_fs))

    # Simplify the up/down ratio
    g = gcd(fs, new_fs)
    up = new_fs // g
    down = fs // g

    # Polyphase resampling with anti-aliasing
    rs = resample_poly(signal, up, down)
    rs = np.asarray(rs).flatten()
    return rs

def preprocess_e4(
    file: str,
    resample_data: bool = False,
    resample_fs: int = 64
) -> Union[pd.DataFrame, tuple]:
    """
    Pre-process electrodermal and temperature data from Empatica E4 files,
    including comma-separated values (.csv) or archive (.zip) files.

    Parameters
    ----------
    file : str
        The path of the Empatica E4 CSV or archive file. The file extension
        must be either '.csv' or '.zip.'
    resample_data : bool, optional
        Whether the EDA and temperature data should be resampled; by
        default, False.
    resample_fs : int, optional
        The new sampling rate to which the data should be resampled; by
        default, 64 Hz.

    Returns
    -------
    pandas.DataFrame or tuple of pandas.DataFrame
        If the input file is an Empatica E4 CSV file, returns a single data
        frame containing the pre-preprocessed data with timestamps.
        If the input file is an Empatica E4 archive file, returns a tuple
        containing two DataFrames:
        - eda_data : pandas.DataFrame
            A DataFrame containing the pre-processed EDA data.
        - temp_data : pandas.DataFrame
            A DataFrame containing the pre-processed temperature data.
    """
    if not file.lower().endswith(('csv', 'zip')):
        raise TypeError('The input filename must end in either \'.csv\' or '
                        '\'.zip\'.')
    else:
        # Pre-process Empatica E4 CSV files
        if file.lower().endswith('csv'):
            meta = pd.read_csv(file, nrows = 2, header = None)
            fs = meta.iloc[1, 0]
            start_time = meta.iloc[0, 0]
            data = pd.read_csv(file, header = 1, names = ['uS'])
            timestamps = pd.date_range(
                start = pd.to_datetime(start_time, unit = 's'),
                periods = len(data), freq = f'{1 / fs}S')
            if resample_data:
                data = pd.Series(resample(data, fs, resample_fs), name = 'uS')
                timestamps = pd.date_range(
                    start = pd.to_datetime(start_time, unit = 's'),
                    periods = len(data), freq = f'{1 / resample_fs}S')
            timestamps = pd.Series(timestamps, name = 'Timestamp')
            e4 = pd.concat([timestamps, data], axis = 1)
            return e4

        # Pre-process Empatica E4 archive files
        else:
            with ZipFile(file) as z:
                if 'EDA.csv' not in z.namelist():
                    raise FileNotFoundError('\'EDA.csv\' file not found.')
                else:
                    eda_file = z.open('EDA.csv')
                    eda = pd.read_csv(eda_file, header = 1, names = ['uS'])
                    eda_file.seek(0)
                    meta = pd.read_csv(eda_file, nrows = 2, header = None)
                    fs = meta.iloc[1, 0]
                    start_time = meta.iloc[0, 0]
                    timestamps = pd.date_range(
                        start = pd.to_datetime(start_time, unit = 's'),
                        periods = len(eda), freq = f'{1 / fs}S')
                    if resample_data:
                        eda = pd.Series(
                            resample(eda, fs, resample_fs), name = 'uS')
                        timestamps = pd.date_range(
                            start = pd.to_datetime(start_time, unit = 's'),
                            periods = len(eda), freq = f'{1 / resample_fs}S')
                    timestamps = pd.Series(timestamps, name = 'Timestamp')
                    eda_data = pd.concat([timestamps, eda], axis = 1)

                if 'TEMP.csv' not in z.namelist():
                    raise FileNotFoundError('\'TEMP.csv\' file not found.')
                else:
                    temp_file = z.open('TEMP.csv')
                    temp = pd.read_csv(
                        temp_file, header = 1, names = ['Celsius'])
                    temp_file.seek(0)
                    meta = pd.read_csv(
                        temp_file, nrows = 2, header = None)
                    fs = meta.iloc[1, 0]
                    start_time = meta.iloc[0, 0]
                    timestamps = pd.date_range(
                        start = pd.to_datetime(start_time, unit = 's'),
                        periods = len(temp), freq = f'{1 / fs}S')
                    if resample_data:
                        temp = pd.Series(
                            resample(temp, fs, resample_fs), name = 'Celsius')
                        timestamps = pd.date_range(
                            start = pd.to_datetime(start_time, unit = 's'),
                            periods = len(temp), freq = f'{1 / resample_fs}S')
                    timestamps = pd.Series(timestamps, name = 'Timestamp')
                    temp_data = pd.concat([timestamps, temp], axis = 1)
            return eda_data, temp_data
        
def _cvxEDA(
    signal: np.ndarray,
    fs: int = 4,
    tau0: float = 2.,
    tau1: float = 0.7,
    delta_knot: float = 10.,
    alpha: float = 8e-4,
    gamma: float = 1e-2,
    solver: 'QP solver' = None,
    options: dict = {'reltol': 1e-9, 'show_progress': True}
) -> tuple:
    """
    Decompose an EDA signal into its phasic and tonic components using the
    convex optimization approach by Greco et al. (2015).

    Parameters
    ----------
    signal : array_like
        An array containing the EDA signal.
    fs : int, optional
        The sampling rate of the EDA signal; by default, 4 Hz.
    tau0 : float, optional
        Slow time constant of the Bateman function; by default, 2.0.
    tau1 : float, optional
        Fast time constant of the Bateman function; by default, 0.7.
    delta_knot : float, optional
        Time between knots of the tonic spline function; by default, 10.0.
    alpha : float, optional
        Penalization for the sparse SMNA driver; by default, 8e-4.
    gamma : float, optional
        Penalization for the tonic spline coefficients; by default, 1e-2.
    solver : object, optional
        Sparse QP solver to be used.
    options : dict, optional
        Solver options.

    Returns
    -------
    r : array_like
        The phasic component.
    p : array_like
        Sparse SMNA driver of phasic component.
    t : array_like
        The tonic component.
    l : array_like
        Coefficients of tonic spline.
    d : array_like
        The Offset and slope of the linear drift term.
    e : array_like
        Model residuals.
    obj : float
        Value of objective function being minimized (equation 15 in paper).

    Notes
    -----
    Copyright (C) 2014-2015 Luca Citi, Alberto Greco

    This program is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 3 of the License, or (at your
    option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details.

    You may contact the original author by e-mail (lciti@ieee.org).

    If you use this program in support of published research, please include a
    citation of the reference below. If you use this code in a software
    package, please explicitly inform end users of this copyright notice and
    ask them to cite the reference above in their published research.

    References
    ----------
    A. Greco, G. Valenza, A. Lanata, E. P. Scilingo, & L. Citi. (2015). cvxEDA:
    A convex optimization approach to electrodermal activity processing. IEEE
    Transactions on Biomedical Engineering, 63(4): 797-804.
    """

    n = len(signal)
    y = cv.matrix(signal)
    delta = 1. / fs

    # Bateman ARMA model
    a1 = 1. / min(tau1, tau0)  # a1 > a0
    a0 = 1. / max(tau1, tau0)
    ar = np.array(
        [(a1 * delta + 2.) * (a0 * delta + 2.), 2. * a1 * a0 * delta ** 2 - 8.,
         (a1 * delta - 2.) * (a0 * delta - 2.)]) / ((a1 - a0) * delta ** 2)
    ma = np.array([1., 2., 1.])

    # Matrices for ARMA model
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n - 2, 1)), np.c_[i, i, i],
                    np.c_[i, i - 1, i - 2], (n, n))
    M = cv.spmatrix(np.tile(ma, (n - 2, 1)), np.c_[i, i, i],
                    np.c_[i, i - 1, i - 2], (n, n))

    # Spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1., delta_knot_s),
                np.arange(delta_knot_s, 0., -1.)]  # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)

    # Matrix of spline regressors
    i = (np.c_[np.arange(-(len(spl) // 2), (len(spl) + 1) // 2)] +
         np.r_[np.arange(0, n, delta_knot_s)])
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl), 1))
    p = np.tile(spl, (nB, 1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # Trend
    C = cv.matrix(np.c_[np.ones(n), np.arange(1., n + 1.) / n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m, n: cv.spmatrix([], [], [], (m, n))
        G = cv.sparse([
            [-A, z(2, n), M, z(nB + 2, n)], [z(n + 2, nC), C, z(nB + 2, nC)],
            [z(n, 1), -1, 1, z(n + nB + 2, 1)],
            [z(2 * n + 2, 1), -1, 1, z(nB, 1)],
            [z(n + 2, nB), B, z(2, nB),
             cv.spmatrix(1.0, range(nB), range(nB))]])
        h = cv.matrix([z(n, 1), .5, .5, y, .5, .5, z(nB, 1)])
        c = cv.matrix(
            [(cv.matrix(alpha, (1, n)) * A).T, z(nC, 1), 1, gamma, z(nB, 1)])
        res = cv.solvers.conelp(c, G, h, dims = {
            'l': n,
            'q': [n + 2, nB + 2],
            's': []
        })
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([
            [Mt * M, Ct * M, Bt * M], [Mt * C, Ct * C, Bt * C],
            [Mt * B, Ct * B,
             Bt * B + gamma * cv.spmatrix(1.0, range(nB), range(nB))]
        ])
        f = cv.matrix(
            [(cv.matrix(alpha, (1, n)) * A).T - Mt * y, -(Ct * y), -(Bt * y)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n, len(f))),
                            cv.matrix(0., (n, 1)), solver = solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n:n + nC]
    t = B * l + C * d
    q = res['x'][:n]
    p = A * q
    r = M * q
    e = y - r - t
    return tuple(np.array(a).ravel() for a in (r, p, t, l, d, e, obj))