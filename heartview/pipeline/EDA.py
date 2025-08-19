import numpy as np
import pandas as pd
import cvxopt as cv
from tqdm import tqdm
from zipfile import ZipFile
from scipy.signal import butter, filtfilt, windows
from scipy.signal import resample as scipy_resample

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

    def bandpass(self, signal, lowcut = 0.05, highcut = 2, order = 2):
        """
        Filter an EDA signal with a Butterworth bandpass filter.

        Parameters
        ----------
        signal : array_like
            An array containing the raw EDA signal.
        lowcut : int, float
            The cut-off frequency at which frequencies below this value in the
            EDA signal are attenuated; by default, 0.05 Hz.
        highcut : int, float
            The cut-off frequency at which frequencies above this value in the
            EDA signal are attenuated; by default, 2.0 Hz.
        order : int
            The filter order, i.e., the number of samples required to produce
            the desired filtered output; by default, 2.

        Returns
        -------
        filtered : array_like
            An array containing the filtered EDA signal.

        Notes
        -----
        This is the all-in-one default filter used by the PhysioView
        dashboard to remove baseline drift, electromyographic (EMG)
        activity, and powerline interference.
        """
        nyq = 0.5 * self.fs
        
        if highcut >= nyq:
            raise ValueError(f'highcut must be less than Nyquist frequency ({nyq} Hz).')
            
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype = 'band')
        filtered = filtfilt(b, a, signal)
        return filtered

    def gaussian(self, signal, window = 40, sigma = 0.4):
        """
        Apply a Gaussian low-pass filter to an EDA signal.

        Parameters
        ----------
        signal : array_like
            An array containing the raw EDA signal.
        window : int
            The number of points in the window; by default, 40.
        sigma : float
            The standard deviation (sigma) of the Gaussian window, in seconds;
            by default, 0.4 s (or 400 ms).

        Returns
        -------
        filtered : array_like
            An array containing the smoothed EDA signal.
            
        References
        ----------
        Campanella, S., Altaleb, A., Belli, A., Pierleoni, P., & Palma, L. 
        (2023). A method for stress detection using empatica E4 bracelet and 
        machine-learning techniques. Sensors (Basel), 23(7), 3565.
        """
        sigma_points = int(sigma * self.fs)
        gaussian_window = windows.gaussian(int(window), sigma_points)
        gaussian_window /= np.sum(gaussian_window)  # normalize
        filtered = np.convolve(signal, gaussian_window, mode = 'same')
        return filtered

    def lowpass(self, signal, cutoff = 2, order = 3):
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
        ma = np.convolve(signal, kernel, mode = 'valid')
        epsilon = np.finfo(float).eps   # pad the beginning with epsilons
        ma = np.concatenate((ma, np.full(len(signal) - len(ma), epsilon)))
        return ma

# ====================== Other EDA Data Pre-Processing =======================
def get_phasic_tonic(signal, fs, show_progress = True):
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

def resample(signal, fs, new_fs):
    """
    Resample electrodermal activity data to a new sampling frequency using
    Fast Fourier Transform (FFT) and interpolation.

    Parameters
    ----------
    signal : array_like
        An array containing the EDA signal.
    fs : int, float
        The sampling rate of the EDA signal.
    new_fs : int, float
        The new sampling rate to which the signal should be resampled.

    Returns
    -------
    rs : array_like
        An array containing the resampled EDA signal.
    """
    resampling_factor = int(new_fs // fs)
    resampled = scipy_resample(signal, len(signal) * resampling_factor)
    resampled = np.array(resampled).flatten()
    return resampled

def segment_data(resampled, original, seg_size = 60, ts_col = None):
    """
    Segment electrodermal activity (EDA) data into windows for further
    processing.

    Parameters
    ----------
    resampled : array_like
        An array containing the resampled EDA data.
    original : pandas.DataFrame
        The original DataFrame containing the raw EDA data.
    seg_size : int
        The size of the segment, in seconds; by default, 60.
    ts_col : str, optional
        The name of the column containing timestamps; by default, None.
        If a string value is given, the output will contain a timestamps
        column.
    """
    segmented = pd.DataFrame()
    return segmented

def preprocess_e4(file, resample_data = False, resample_fs = 64):
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
        
def _cvxEDA(signal, fs, tau0 = 2., tau1 = 0.7, delta_knot = 10., alpha = 8e-4,
           gamma = 1e-2, solver = None,
           options = {'reltol': 1e-9, 'show_progress': True}):
    """
    Decompose an EDA signal into its phasic and tonic components using the
    convex optimization approach by Greco et al. (2015).

    Parameters
    ----------
    signal : array_like
        An array containing the EDA signal.
    fs : float
        The sampling rate of the EDA signal.
    tau0 : float
        Slow time constant of the Bateman function; by default, 2.0.
    tau1 : float
        Fast time constant of the Bateman function; by default, 0.7.
    delta_knot : float
        Time between knots of the tonic spline function; by default, 10.0.
    alpha : float
        Penalization for the sparse SMNA driver; by default, 8e-4.
    gamma : float
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
    A Greco, G. Valenza, A. Lanata, E. P. Scilingo, & L. Citi. (2015). cvxEDA:
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
    spl = np.r_[np.arange(1., delta_knot_s), np.arange(delta_knot_s, 0.,
                                                       -1.)]  # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # Matrix of spline regressors
    i = np.c_[np.arange(-(len(spl) // 2), (len(spl) + 1) // 2)] + np.r_[
        np.arange(0, n, delta_knot_s)]
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
    return (np.array(a).ravel() for a in (r, p, t, l, d, e, obj))