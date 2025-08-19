from zipfile import ZipFile, ZipExtFile
from tqdm import tqdm
from scipy.signal import resample as scipy_resample
from plotly.subplots import make_subplots
from heartview.pipeline.ACC import compute_magnitude
from heartview.pipeline.SQA import Cardio, EDA
from heartview.pipeline.PPG import BeatDetectors
from heartview.pipeline.EDA import Filters as edaFilters
import warnings
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import pyedflib

class Actiwave:
    """
    A class for convenient pre-processing of data from the Actiwave Cardio
    device.

    Parameters/Attributes
    ---------------------
    file : str
        The path of the Actiwave Cardio device file saved in European
        Data Format (.edf).
    """

    def __init__(self, file):
        """
        Initialize the Actiwave object.

        Parameters
        ----------
        file : str
            The path of the Actiwave Cardio device file saved in European
            Data Format (.edf).
        """
        if not file.endswith(('.edf', '.EDF')):
            raise ValueError(
                'Invalid file path. The `file` parameter must take a string '
                'value ending in \'.EDF\' or \'.edf\'.')
        else:
            self.file = file

    def preprocess(self, time_aligned = False):
        """
        Pre-process electrocardiograph (ECG) and acceleration data from
        an Actiwave Cardio file.

        Parameters
        ----------
        time_aligned : bool, optional
            Whether to time-align ECG and acceleration data based on the
            sampling rate of the ECG data; by default, False.

        Returns
        -------
        tuple or pandas.DataFrame
            If `time_aligned` is False, returns a tuple (`ecg`, `acc`),
            where `ecg` is a DataFrame containing the pre-processed ECG data
            and `acc` is a DataFrame containing the pre-processed X-, Y-, and
            Z-axis acceleration data. If `time_aligned` is True, returns a
            single DataFrame containing time-synced ECG and acceleration
            data according to the ECG data's timestamps.
        """
        f = pyedflib.EdfReader(self.file)
        start = dt.datetime.timestamp(f.getStartdatetime())
        end = start + f.getFileDuration()
        ecg, acc = pd.DataFrame(), pd.DataFrame()
        signal_labels = f.getSignalLabels()
        ecg_chn = [i for i in range(len(signal_labels))
                   if 'ECG' in signal_labels[i]]
        acc_chn = [i for i in range(len(signal_labels))
                   if 'X' in signal_labels[i]
                   or 'Y' in signal_labels[i]
                   or 'Z' in signal_labels[i]]
        acc_sig = dict(zip(['X', 'Y', 'Z'], acc_chn))
        ecg_fs = f.getSampleFrequency(ecg_chn[0])
        acc_fs = f.getSampleFrequency(acc_chn[0])

        # Get ECG data
        ecg['Timestamp'] = np.arange(start, end, (1 / ecg_fs))
        ecg['ECG'] = pd.Series(f.readSignal(ecg_chn[0]) / 1000)
        ecg['Timestamp'] = ecg['Timestamp'].apply(
            lambda t: dt.datetime.utcfromtimestamp(t))

        # Get ACC data
        acc['Timestamp'] = np.arange(start, end, (1 / acc_fs))
        for k, v in acc_sig.items():
            acc[k] = pd.Series(f.readSignal(v))
        acc['Magnitude'] = np.sqrt(acc[['X', 'Y', 'Z']].apply(
            lambda x: x ** 2).sum(axis = 1))
        acc['Timestamp'] = acc['Timestamp'].apply(
            lambda t: dt.datetime.utcfromtimestamp(t))
        f.close()

        if time_aligned:
            resampled = pd.DataFrame()
            for col in ['X', 'Y', 'Z']:
                rs = scipy_resample(acc[col], len(ecg))
                resampled = pd.concat(
                    [resampled, pd.Series(rs, name = col)], axis = 1)
            preprocessed = pd.concat([ecg, resampled], axis = 1)
            return preprocessed
        else:
            return ecg, acc

    def get_ecg_fs(self):
        """
        Get the sampling rate of ECG data from an Actiwave Cardio device.

        Returns
        -------
        fs : int, float
            The sampling rate of the ECG recording.
        """
        f = pyedflib.EdfReader(self.file)
        signal_labels = f.getSignalLabels()
        for chn in range(len(signal_labels)):
            if 'ECG' in signal_labels[chn]:
                ecg_chn = chn
        try:
            fs = f.getSampleFrequency(ecg_chn)
            return fs
        except NameError:
            raise NameError('No ECG channel found.')
        finally:
            f.close()

    def get_acc_fs(self):
        """
        Get the sampling rate of accelerometer data from an Actiwave Cardio
        device.

        Returns
        -------
        fs : int, float
            The sampling rate of the accelerometer recording.
        """
        f = pyedflib.EdfReader(self.file)
        signal_labels = f.getSignalLabels()
        for chn in range(len(signal_labels)):
            if 'X' in signal_labels[chn]:
                acc_chn = chn
        try:
            fs = f.getSampleFrequency(acc_chn)
            return fs
        except NameError:
            raise NameError('No ACC channels found.')
        finally:
            f.close()

# ==================== Empatica E4 Pre-Processing and SQA ====================
class Empatica:
    """
    A class to conveniently pre-process and assess quality of PPG and EDA data
    from Empatica E4 devices.

    Attributes
    ----------
    file : str
        The path of the Empatica archive file with a '.zip' extension.
    """

    class Data:
        """A class to store pre-processed data variables."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __init__(self, file):
        """
        Initialize the Empatica object.

        Parameters
        ----------
        file : str
            The path of the Empatica archive file with a '.zip' extension.
        """
        if not file.endswith(('.zip', '.ZIP')):
            raise ValueError(
                'Invalid file path. The `file` parameter must take a string '
                'value ending in \'.zip\' or \'.ZIP\'.')
        else:
            self.file = file

    def preprocess(self, time_aligned = False):
        """
        Pre-process all data from the Empatica E4.

        Parameters
        ----------
        time_aligned : bool, optional
            Whether to time-align all data based on the signal with the
            highest sampling rate (i.e. blood volume pulse); by default,
            False.

        Returns
        -------
        data : Empatica.Data object
            An `Empatica.Data` object with the following attributes and
            corresponding pre-processed data:

            If `time_aligned` is False:
                acc : pandas.DataFrame
                    A DataFrame containing the pre-processed ACC data with
                    corresponding timestamps.
                bvp : pandas.DataFrame
                    A DataFrame containing the pre-processed BVP data with
                    corresponding timestamps.
                eda : pandas.DataFrame
                    A DataFrame containing the pre-processed EDA data with
                    corresponding timestamps.
                hr : pandas.DataFrame
                    A DataFrame containing the pre-processed HR data with
                    corresponding timestamps.
                ibi : pandas.DataFrame
                    A DataFrame containing the pre-processed IBI data with
                    corresponding timestamps and seconds elapsed since the
                    start time of the IBI recording.
                temp : pandas.DataFrame
                    A DataFrame containing the pre-processed temperature
                    data with corresponding timestamps.
                start_time : float
                    The Unix-formatted start time of the E4 recording.
                bvp_fs : float
                    The sampling rate of the BVP recording.
                eda_fs : float
                    The sampling rate of the EDA recording.

            If `time_aligned` is True:
                hrv : pandas.DataFrame
                    A DataFrame containing time-synced BVP, HR, IBI,
                    and acceleration data.
                eda : pandas.DataFrame
                    A DataFrame containing time-synced EDA, temperature,
                    and acceleration data.
                start_time : float
                    The Unix-formatted start time of the E4 recording.
                bvp_fs : float
                    The sampling rate of the BVP recording.
                eda_fs : float
                    The sampling rate of the EDA recording.

        Examples
        --------
        >>> from heartview import heartview
        >>> e4_archive = 'Sample_E4_Data.zip'
        >>> E4 = heartview.Empatica(e4_archive)
        >>> ALL_E4_DATA = E4.preprocess()
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            for file in e4_files:
                if 'ACC' in file:
                    with archive.open(file) as acc_file:
                        acc_data = self.get_acc().acc
                if 'BVP' in file:
                    with archive.open(file) as bvp_file:
                        bvp_data = self.get_bvp().bvp
                        start_time = self.get_bvp().start
                        bvp_fs = self.get_bvp().fs
                if 'EDA' in file:
                    with archive.open(file) as eda_file:
                        eda_data = self.get_eda().eda
                        start_time = self.get_eda().start
                        eda_fs = self.get_eda().fs
                if 'HR' in file:
                    with archive.open(file) as hr_file:
                        hr_data = self.get_hr().hr
                if 'IBI' in file:
                    with archive.open(file) as ibi_file:
                        ibi_data = self.get_ibi().ibi
                if 'TEMP' in file:
                    with archive.open(file) as temp_file:
                        temp_data = self.get_temp().temp

        if time_aligned:
            # Merge IBI and HR values into BVP data frame
            full_hrv = pd.merge_asof(
                bvp_data, ibi_data.drop(['Seconds'], axis = 1),
                on = 'Timestamp', direction = 'nearest')
            full_hrv = pd.merge_asof(
                full_hrv, hr_data,
                on = 'Timestamp', direction = 'nearest')
            bvp_ts = bvp_data['Timestamp'].values
            ibi_ts = ibi_data['Timestamp'].values
            hr_ts = hr_data['Timestamp'].values
            ibi_insertion_points = np.searchsorted(bvp_ts, ibi_ts) - 1
            hr_insertion_points = np.searchsorted(bvp_ts, hr_ts)
            full_hrv.loc[~np.isin(np.arange(len(full_hrv)),
                                  ibi_insertion_points), 'IBI'] = np.nan
            full_hrv.loc[~np.isin(np.arange(len(full_hrv)),
                                  hr_insertion_points), 'HR'] = np.nan

            # Resample acceleration data to match BVP and EDA sampling rates
            acc_rs = pd.DataFrame()
            acc_cols = ['X', 'Y', 'Z', 'Magnitude']
            for ref_data in [bvp_data, eda_data]:
                acc_rs[acc_cols] = acc_data[acc_cols].apply(
                    lambda a: scipy_resample(a, len(ref_data)))
                if ref_data is bvp_data:
                    full_hrv = pd.merge(full_hrv, acc_rs,
                                        left_index = True, right_index = True)
                else:
                    full_eda = pd.merge(eda_data, temp_data,
                                        on = 'Timestamp', how = 'inner')
                    full_eda = pd.merge(full_eda, acc_rs,
                                        left_index = True, right_index = True)
            data = self.Data(**{'hrv': full_hrv,
                                'eda': full_eda,
                                'start': start_time,
                                'bvp_fs': bvp_fs,
                                'eda_fs': eda_fs})

        else:
            data = self.Data(**{'acc': acc_data,
                                'bvp': bvp_data,
                                'eda': eda_data,
                                'hr': hr_data,
                                'ibi': ibi_data,
                                'temp': temp_data,
                                'start': start_time,
                                'bvp_fs': bvp_fs,
                                'eda_fs': eda_fs})
        return data

    def get_acc(self):
        """
        Get the pre-processed acceleration data and its start time and
        sampling rate from the Empatica E4.

        Returns
        -------
        acc_data : Empatica.Data object
            An `Empatica.Data` object with the following attributes and
            corresponding accelerometer data variables:

            acc : pandas.DataFrame
                A DataFrame containing the pre-processed BVP data with
                corresponding timestamps.
            start : float
                The Unix-formatted start time of the BVP recording.
            fs : int
                The sampling rate of the BVP data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            acc_file = None
            for file in e4_files:
                if 'ACC' in file:
                    acc_file = file
                    break
            if acc_file is None:
                raise ValueError('No "ACC.csv" file found.')
            with archive.open(file) as acc_file:
                acc, acc_start, acc_fs = self._get_e4_data(
                    acc_file, name = ['X', 'Y', 'Z'])
                acc = acc.apply(lambda x: (x / 64) * 9.81
                                if x.name != 'Timestamp' else x)
                acc['Magnitude'] = compute_magnitude(
                    acc['X'], acc['Y'], acc['Z'])
            acc_data = self.Data(**{'acc': acc,
                                    'start': acc_start,
                                    'fs': acc_fs})
            return acc_data

    def get_bvp(self):
        """
        Get the raw blood volume pulse (BVP) data and its start time and
        sampling rate from the Empatica E4.

        Returns
        -------
        bvp_data : Empatica.Data object
            An `Empatica.Data` object with the following attributes and
            corresponding BVP data variables:

            bvp : pandas.DataFrame
                A DataFrame containing the pre-processed BVP data with
                corresponding timestamps.
            start : float
                The Unix-formatted start time of the BVP recording.
            fs : int
                The sampling rate of the BVP data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            bvp_file = None
            for file in e4_files:
                if 'BVP' in file:
                    bvp_file = file
                    break
            if bvp_file is None:
                raise ValueError('No "BVP.csv" file found.')
            with archive.open(bvp_file) as bvp_file:
                bvp, bvp_start, bvp_fs = self._get_e4_data(
                    bvp_file, name = 'BVP')
            bvp_data = self.Data(**{'bvp': bvp,
                                    'start': bvp_start,
                                    'fs': bvp_fs})
            return bvp_data

    def get_eda(self):
        """
        Get the raw electrodermal activity (EDA) data and its recording
        start time and sampling rate from the Empatica E4.

        Returns
        -------
        eda_data : Empatica.Data object
            An `Empatica.Data` object with the following attributes and
            corresponding EDA data variables:

            eda : pandas.DataFrame
                A DataFrame containing the pre-processed EDA data with
                corresponding timestamps.
            start : float
                The Unix-formatted start time of the EDA recording.
            fs : int
                The sampling rate of the EDA data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            eda_file = None
            for file in e4_files:
                if 'EDA' in file:
                    eda_file = file
                    break
            if eda_file is None:
                raise ValueError('No "EDA.csv" file found.')
            with archive.open(eda_file) as eda_file:
                eda, eda_start, eda_fs = self._get_e4_data(
                    eda_file, name = 'EDA')
            eda_data = self.Data(**{'eda': eda,
                                    'start': eda_start,
                                    'fs': eda_fs})
            return eda_data

    def get_hr(self):
        """
        Get the pre-processed heart rate (HR) data, start time of the
        first HR measurement, and sampling rate from the Empatica E4.

        Returns
        -------
        hr_data : Empatica.Data object
            An `Empatica.Data` object with the following attributes and
            corresponding HR data variables:

            hr : pandas.DataFrame
                A DataFrame containing the pre-processed HR data with
                corresponding timestamps.
            start : float
                The Unix-formatted start time of the HR measurements.
            fs : int
                The sampling rate of the BVP data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            hr_file = None
            for file in e4_files:
                if 'HR' in file:
                    hr_file = file
                    break
            if hr_file is None:
                raise ValueError('No "HR.csv" file found.')
            with archive.open(file) as hr_file:
                hr, hr_start, hr_fs = self._get_e4_data(
                    hr_file, name = 'HR')
            hr_data = self.Data(**{'hr': hr,
                                   'start': hr_start,
                                   'fs': hr_fs})
            return hr_data

    def get_ibi(self):
        """
        Get the pre-processed interbeat interval (IBI) data and the start
        time of the first interval from the Empatica E4.

        Returns
        -------
        ibi_data : Empatica.Data object
            An `Empatica.Data` object with the following attributes and
            corresponding IBI data variables:

            ibi : pandas.DataFrame
                A DataFrame containing the pre-processed IBI data with
                corresponding timestamps.
            start : int
                The Unix-formatted start time of the IBI data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            ibi_file = None
            for file in e4_files:
                if 'IBI' in file:
                    ibi_file = file
                    break
            if ibi_file is None:
                raise ValueError('No "IBI.csv" file found.')
            with archive.open(file) as ibi_file:
                ibi = pd.read_csv(ibi_file, header = 0,
                                  names = ['Seconds', 'IBI'])
                ibi_file.seek(0)
                ibi_start = self._get_e4_start_time(ibi_file)
                ibi['IBI'] *= 1000
                ibi.insert(
                    0, 'Timestamp', (ibi['Seconds'] + ibi_start).apply(
                        lambda t: dt.datetime.utcfromtimestamp(t)))
            ibi_data = self.Data(**{'ibi': ibi, 'start': ibi_start})
            return ibi_data

    def get_temp(self):
        """
        Get the raw skin temperature data and its recording start time and
        sampling rate from the Empatica E4.

        Returns
        -------
        temp_data : Empatica.Data object
            An `Empatica.Data` object with the following attributes and
            corresponding temperature data variables:

            temp : pandas.DataFrame
                A DataFrame containing the pre-processed temperature data with
                corresponding timestamps.
            start : float
                The Unix-formatted start time of the temperature recording.
            fs : int
                The sampling rate of the temperature data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            temp_file = None
            for file in e4_files:
                if 'TEMP' in file:
                    temp_file = file
                    break
            if temp_file is None:
                raise ValueError('No "TEMP.csv" file found.')
            with archive.open(temp_file) as temp_file:
                temp, temp_start, temp_fs = self._get_e4_data(
                    temp_file, name = 'Temp')
            temp_data = self.Data(**{'temp': temp,
                                     'start': temp_start,
                                     'fs': temp_fs})
            return temp_data

    def get_e4_beats(self, bvp_data, ibi_data, start_time,
                     show_progress = True):
        """
        Get locations of beats from Empatica E4 interbeat interval (IBI)
        data relative to its blood volume pulse (BVP) data.

        Parameters
        ----------
        bvp_data : pandas.DataFrame
            A DataFrame containing the Empatica E4 BVP data, outputted from
            `Empatica.preprocess()`.
        ibi_data : pandas.DataFrame
            A DataFrame containing the Empatica E4 IBI data, outputted from
            `Empatica.preprocess()`.
        start_time : int
            The Unix timestamp of the recording start time.
        show_progress : bool, optional
            Whether to display a progress bar while the function runs; by
            default, True.

        Returns
        -------
        e4_beats : list
            A list containing the indices of beats extracted from IBI data of
            the Empatica E4.
        """
        ibi = ibi_data.copy()
        bvp = bvp_data.copy()
        ibi['Unix'] = ibi['Seconds'] + start_time
        ibi['Timestamp'] = ibi['Unix'].apply(
            lambda t: dt.datetime.utcfromtimestamp(t))
        bvp['Timestamp'] = pd.to_datetime(bvp['Timestamp'])
        e4_beats = []
        for t in tqdm(ibi['Timestamp'], disable = not show_progress):
            time_diff = np.abs(bvp['Timestamp'] - t)
            closest_ix = time_diff.idxmin()
            e4_beats.append(closest_ix)
        return e4_beats

    def compute_sqa(self, kind, seg_size = 60, initial_hr = 'auto',
                    min_hr = 40, min_eda = 0.05, max_eda = 40,
                    rolling_window = None, rolling_step = 15,
                    show_progress = True):
        """
        Compute signal quality assessment metrics (SQA) for PPG and/or EDA
        data from Empatica E4 devices.

        Parameters
        ----------
        kind : str
            The kind of data whose SQA to compute. This value must be a string
            variation of 'all', 'eda', or 'ppg'.
        seg_size : int
            The segment size in seconds; by default, 60.
        initial_hr : int, float, or 'auto', optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, 'auto' for automatic calculation
            using the mean heart rate value obtained from six consecutive
            IBIs with the smallest average successive difference.
        min_hr : int, float
            The minimum acceptable heart rate against which the number of
            beats in the last partial segment will be compared; by default, 40.
        min_eda : float, optional
            The minimum acceptable value for EDA data in microsiemens; by
            default, 0.05 uS.
        max_eda : float, optional
            The maximum acceptable value for EDA data in microsiemens; by
            default, 40 uS.
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
        """

        if kind.lower() not in ('all', 'eda', 'ppg'):
            raise ValueError('The `kind` parameter must take a string value '
                             '\'all\', \'eda\', or \'ppg\'.')
        else:
            if kind == 'all':
                kind = ('eda', 'ppg')

            ppg_metrics, eda_metrics = None, None

            if 'ppg' in kind:
                bvp = self.get_bvp().bvp
                bvp_start = self.get_bvp().start
                fs = self.get_bvp().fs
                ibi = self.get_ibi().ibi
                ppg_beats = BeatDetectors(fs, False).adaptive_threshold(
                    bvp['BVP'])
                sqa = Cardio(fs)
                artifact_beats = sqa.identify_artifacts(
                    ppg_beats, 'both', initial_hr, 6, 5, 1)
                ppg_metrics = sqa.compute_metrics(
                    bvp, ppg_beats, artifact_beats, 'Timestamp', seg_size,
                    min_hr, rolling_window, rolling_step, show_progress)
            if 'eda' in kind:
                eda = self.get_eda().eda
                eda_start = self.get_eda().start
                fs = self.get_eda().fs
                eda['EDA'] = edaFilters(fs).gaussian(
                    eda['EDA'], window = int(0.5 * fs))
                temp = self.get_temp().temp
                sqa = EDA(fs, eda_min = min_eda, eda_max = max_eda)
                eda_metrics = sqa.compute_metrics(
                    eda['EDA'], temp['Temp'], eda['Timestamp'],
                    preprocessed = True, seg_size = 60,
                    rolling_window = rolling_window,
                    rolling_step = rolling_step)
        if ppg_metrics is not None and eda_metrics is not None:
            return ppg_metrics, eda_metrics
        else:
            if ppg_metrics is not None:
                return ppg_metrics
            if eda_metrics is not None:
                return eda_metrics

    def plot_signals(self, segment = 1, seg_size = 60, interactive = True):
        """
        Display a plot of a segment of signals recorded with the Empatica E4
        device.

        Parameters
        ----------
        segment : int, optional
            The number of the position of the segment to plot; by default, 1.
        seg_size : int, optional
            The segment size in seconds; by default, 60.
        interactive : bool, optional
            Whether to plot an interactive visualization; by default, True.

        Returns
        -------
        fig : plotly.graph_objects.Figure or None
            If `interactive` is True, displays and returns an interactive
            Plotly figure containing the plotted signals. If `interactive`
            is False, displays a static figure and returns None.
        """
        data = self.preprocess()
        dtypes = ('acc', 'bvp', 'eda', 'temp')
        if interactive:
            fig = make_subplots(
                rows = 4, cols = 1,
                shared_xaxes = True,
                vertical_spacing = 0.02,
                row_heights = [0.2, 0.3, 0.3, 0.2])
            for n in range(len(dtypes)):
                if dtypes[n] in ('acc', 'bvp'):
                    fs = data.bvp_fs
                    seg_start = int((segment - 1) * fs * seg_size)
                    seg_end = seg_start + int(fs * seg_size)
                    signal_name = 'ACC' if dtypes[n] == 'acc' else 'BVP'
                    color = 'forestgreen' if dtypes[n] == 'acc' else '#3562bd'
                    ylabel = 'm/s²' if dtypes[n] == 'acc' else ''
                    if dtypes[n] == 'acc':
                        x = data.acc['Timestamp'].iloc[seg_start:seg_end]
                        y = data.acc['Magnitude'].iloc[seg_start:seg_end]
                    else:
                        x = data.bvp['Timestamp'].iloc[seg_start:seg_end]
                        y = data.bvp['BVP'].iloc[seg_start:seg_end]
                else:
                    fs = data.eda_fs
                    seg_start = int((segment - 1) * fs * seg_size)
                    seg_end = seg_start + int(fs * seg_size)
                    signal_name = 'EDA' if dtypes[n] == 'eda' else 'Temperature'
                    color = '#43c9de' if dtypes[n] == 'eda' else '#8b3ac9'
                    ylabel = 'uS' if dtypes[n] == 'eda' else '°C'
                    if dtypes[n] == 'eda':
                        x = data.eda['Timestamp'].iloc[seg_start:seg_end]
                        y = data.eda['EDA'].iloc[seg_start:seg_end]
                    else:
                        x = data.temp['Timestamp'].iloc[seg_start:seg_end]
                        y = data.temp['Temp'].iloc[seg_start:seg_end]
                fig.add_trace(
                    go.Scatter(
                        x = x, y = y,
                        name = signal_name,
                        line = dict(color = color, width = 1.5),
                        hovertemplate = f'<b>{signal_name}</b>: %{{y:.2f}} '
                                        f'{ylabel}<extra></extra>'),
                    row = n+1, col = 1)
                fig.update_yaxes(
                    title_text = ylabel,
                    row = 1, col = 1,
                    showgrid = True,
                    gridwidth = 0.5,
                    gridcolor = 'lightgrey',
                    griddash = 'dot',
                    tickcolor = 'grey',
                    linecolor = 'grey')
            fig.show()
            return fig
        else:
            fig, axs = plt.subplots(4, 1, figsize = (10, 8))
            for n in range(len(dtypes)):
                fs = data.eda_fs
                seg_start = int((segment - 1) * fs * seg_size)
                seg_end = seg_start + int(fs * seg_size)
                if dtypes[n] in ('acc', 'bvp'):
                    signal_name = 'ACC' if dtypes[n] == 'acc' else 'BVP'
                    color = 'forestgreen' if dtypes[n] == 'acc' else '#3562bd'
                    ylabel = 'm/s²' if dtypes[n] == 'acc' else 'BVP'
                    if dtypes[n] == 'acc':
                        x = data.acc['Timestamp'].iloc[seg_start:seg_end]
                        y = data.acc['Magnitude'].iloc[seg_start:seg_end]
                    else:
                        x = data.bvp['Timestamp'].iloc[seg_start:seg_end]
                        y = data.bvp['BVP'].iloc[seg_start:seg_end]
                else:
                    signal_name = 'EDA' if dtypes[n] == 'eda' else 'Temperature'
                    color = '#43c9de' if dtypes[n] == 'eda' else '#8b3ac9'
                    ylabel = 'uS' if dtypes[n] == 'eda' else '°C'
                    if dtypes[n] == 'eda':
                        x = data.eda['Timestamp'].iloc[seg_start:seg_end]
                        y = data.eda['EDA'].iloc[seg_start:seg_end]
                    else:
                        x = data.temp['Timestamp'].iloc[seg_start:seg_end]
                        y = data.temp['Temp'].iloc[seg_start:seg_end]
                for ax in axs:
                    ax.plot(x, y, label = signal_name, color = color, lw = 1.2)
                    ax.set_xlabel('Timestamp')
                    ax.set_ylabel(ylabel)
                    ax.legend(frameon = False)
            plt.tight_layout()
            plt.show()
            return fig, axs

    def _get_e4_data(self, file, name):
        """Extract data from an Empatica E4 file."""
        if not isinstance(name, list) and not isinstance(name, str):
            raise ValueError('The `name` parameter must take either a string '
                             'or a list of strings.')
        else:
            if isinstance(name, list):
                col_name = name
            else:
                col_name = [name]
        data = pd.read_csv(file, header = 1, names = col_name)
        if isinstance(file, str):
            fs = self._get_e4_fs(file)
            start_time = self._get_e4_start_time(file)
        else:
            if hasattr(file, 'seek'):
                file.seek(0)
                fs = self._get_e4_fs(file)
                file.seek(0)
                start_time = self._get_e4_start_time(file)
        timestamps = pd.date_range(
            start = pd.to_datetime(start_time, unit = 's'),
            periods = len(data), freq = f'{1 / fs}S')
        timestamps = pd.Series(timestamps, name = 'Timestamp')
        data = pd.merge(timestamps, data,
                        left_index = True, right_index = True)
        return data, start_time, fs

    def _get_e4_fs(self, file):
        """Get the sampling rate from an Empatica E4 file."""
        contents = pd.read_csv(file, header = None, nrows = 2, usecols = [0])
        fs = contents.iloc[1].item()
        return fs

    def _get_e4_start_time(self, file):
        """Get the Unix-formatted start time of an Empatica E4 recording."""
        contents = pd.read_csv(file, header = None, nrows = 2, usecols = [0])
        if type(file) is ZipExtFile:
            if 'IBI' in file.name:
                start = contents.loc[0, 0]
            else:
                start = contents.iloc[0].item()
        else:
            if file.endswith('IBI.csv'):
                start = contents.loc[0, 0]
            else:
                start = contents.iloc[0].item()
        return start

# ======================== Other Data Pre-Processing =========================
def get_duration(data, fs, unit = 'sec'):
    """
    Get the duration of a signal.

    Parameters
    ----------
    data : array_like
        An array or DataFrame containing the signal.
    fs : int
        The sampling rate of the data.
    unit : str
        The unit in which the duration should be calculated; by default,
        in seconds (`sec`).

    Returns
    -------
    dur : float
        The duration of the signal.
    """

    dur = len(data) / fs
    if unit not in ['sec', 's', 'min', 'm', 'hour', 'h']:
        raise ValueError('The `unit` parameter must take \'sec\', \'min\', '
                         'or \'hour\'.')
    else:
        if unit in ('min', 'm'):
            return round((dur / 60), 2)
        if unit == ('hour', 'h'):
            return round(((dur / 60) / 60), 2)
    return round(dur, 2)

def segment_data(data, fs, seg_size):
    """
    Segment data into specific window sizes.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to be segmented.
    fs : int
        The sampling rate of the data.
    seg_size : int
        The window size, in seconds, into which the data should be
        segmented.

    Returns
    -------
    df : pd.DataFrame
        The original DataFrame with data segmented with labels in a
        'Segment' column.
    """
    df = data.copy()
    df.insert(0, 'Segment', 0)
    segment = 1
    for n in range(0, len(df), int(seg_size * fs)):
        df.loc[n:(n + int(seg_size * fs)), 'Segment'] = segment
        segment += 1
    return df

def compute_ibis(data, fs, beats_ix, ts_col = None):
    """
    Compute interbeat intervals from beat locations in electrocardiograph
     (ECG) or photoplethysmograph (PPG) data.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the pre-processed ECG/PPG data.
    fs : int
        The sampling rate of the ECG/PPG data.
    beats_ix : array_like
        An array of indices corresponding to beat occurrences.
    ts_col : str
        The name of the column in `data` containing timestamp values; by
        default, None.

    Returns
    -------
    ibi : pd.DataFrame
        A DataFrame containing timestamps and IBI values.
    """

    df = data.copy()
    ibis = (np.diff(beats_ix) / fs) * 1000
    if ts_col is not None:
        ibi = df[[ts_col]].copy()
    else:
        ibi = pd.DataFrame({'Sample': np.arange(len(df)) + 1})
    for n, ix in enumerate(beats_ix[1:]):
        ibi.loc[ix, 'IBI'] = ibis[n]
    return ibi

def plot_cardio_signals(signal, fs, ibi, signal_type, x = 'Timestamp',
                        y = 'Filtered', acc = None, seg_num = 1,
                        seg_size = 60, title = None, overlay_corrected = False):
    """
    Create subplots of the electrocardiograph (ECG) or photoplethysmograph
    (PPG), interbeat interval (IBI), and acceleration data (if any).

    Parameters
    ----------
    signal : pandas.DataFrame
        A DataFrame containing the pre-processed ECG or PPG data with beat
        and artifact occurrences in a "Beat" and "Artifact" column.
    fs : int
        The sampling rate of the ECG or PPG data.
    ibi : pandas.DataFrame
        A DataFrame containing IBI values in an "IBI" column.
    signal_type : str
        The type of cardiovascular data being plotted. This must be either
        'ECG' or 'PPG'.
    x : str, optional
        The name of the column of values in the `signal` DataFrame to plot
        along the x-axis; by default, 'Timestamp'.
    y : str, optional
        The column name of values to plot along the y-axis; by default,
        'Filtered'.
    acc : pandas.DataFrame, optional
        A DataFrame containing pre-processed acceleration data with
        magnitude values in a "Magnitude" column.
    seg_num : int
        The segment to plot.
    seg_size : int
        The size of the segment, in seconds; by default, 60.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A figure containing subplots of ECG or PPG data with beat annotations
        and its corresponding IBI data.

    See Also
    --------
    heartview.compute_ibis : Compute IBIs in a DataFrame time-aligned to its
    corresponding cardiovascular data.
    """

    if overlay_corrected and 'Corrected' not in signal.columns:
        raise NotImplementedError('To overlay corrected beats, beat correction must be performed first.')

    seg_start = int((seg_num - 1) * seg_size * fs)
    seg_end = int(seg_start + (fs * seg_size))
    for df in [signal, ibi]:
        df[x] = pd.to_datetime(df[x])
    signal_segment = signal.iloc[seg_start:seg_end]
    ibi_segment = ibi.iloc[seg_start:seg_end].dropna()

    x_array = signal_segment[x]
    if not pd.api.types.is_datetime64_any_dtype(x_array):
        artifact_hover = '<b>Potential Artifact</b> <extra></extra>'
        beat_hover = '<b>Beat</b> <extra></extra>'
    else:
        artifact_hover = '<b>Potential Artifact</b>: %{x|%H:%M:%S.%3f} ' \
                         '<extra></extra>'
        beat_hover = '<b>Beat</b>: %{x|%H:%M:%S.%3f} <extra></extra>'
    if signal_type == 'PPG' or signal_type == 'BVP':
        y_axis = 'bvp'
    else:
        y_axis = 'mV'

    if acc is not None:
        fig = make_subplots(rows = 3, cols = 1,
                            shared_xaxes = True,
                            vertical_spacing = 0.02,
                            row_heights = [0.25, 0.50, 0.25])

        # ACC subplot
        acc = scipy_resample(acc['Magnitude'], len(signal))
        acc_segment = acc[seg_start:seg_end]
        fig.add_trace(
            go.Scatter(
                x = x_array,
                y = acc_segment,
                name = 'ACC',
                line = dict(color = 'forestgreen', width = 1.5),
                hovertemplate = '<b>ACC</b>: %{y:.2f} m/s² <extra></extra>'),
            row = 1, col = 1)
        fig.update_yaxes(
            title_text = 'm/s²',
            title_standoff = 5,
            row = 1, col = 1,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey',
            griddash = 'dot',
            tickcolor = 'grey',
            linecolor = 'grey')

        # ECG/PPG subplot
        fig.add_trace(
            go.Scatter(
                x = x_array,
                y = signal_segment[y],
                name = signal_type,
                showlegend = True,
                line = dict(color = '#3562bd', width = 1.5),
                hovertemplate = f'<b>{signal_type}:</b> %{{y:.2f}} {y_axis} '
                                f'<extra></extra>'),
            row = 2, col = 1)
        fig.update_yaxes(
            title_text = y_axis,
            title_standoff = 5,
            row = 2, col = 1,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey',
            griddash = 'dot',
            tickcolor = 'grey',
            linecolor = 'grey')

        # IBI subplot
        fig.add_trace(
            go.Scatter(
                x = ibi_segment[x],
                y = ibi_segment['IBI'],
                name = 'IBI',
                line = dict(color = '#eb4034', width = 1.5),
                hovertemplate = '<b>IBI</b>: %{y:.2f} ms <extra></extra>'),
            row = 3, col = 1)
        fig.update_yaxes(
            title_text = 'ms',
            row = 3, col = 1,
            title_standoff = 1,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey',
            griddash = 'dot',
            tickcolor = 'grey',
            linecolor = 'grey')

        # Detected beats
        fig.add_trace(
            go.Scatter(
                x = signal_segment.loc[signal_segment.Beat == 1, x],
                y = signal_segment.loc[signal_segment.Beat == 1, y],
                name = 'Detected Beat',
                showlegend = True,
                mode = 'markers',
                marker = dict(color = '#f9c669', size = 6),
                hovertemplate = beat_hover),
            row = 2, col = 1)

        # Artifactual beats
        fig.add_trace(
            go.Scatter(
                x = signal_segment.loc[signal_segment.Artifact == 1, x],
                y = signal_segment.loc[signal_segment.Artifact == 1, y],
                name = 'Potential Artifact',
                showlegend = True,
                mode = 'markers',
                marker = dict(color = 'red'),
                hovertemplate = artifact_hover),
            row = 2, col = 1)

        # Corrected beats
        if overlay_corrected:
            fig.add_trace(
                go.Scatter(
                    x = signal_segment.loc[signal_segment.Corrected == 1, x],
                    y = signal_segment.loc[signal_segment.Corrected == 1, y],
                    name = 'Corrected Beat',
                    showlegend = True,
                    mode = 'markers',
                    marker = dict(color='rgba(250, 250, 250, 0.0)', line = dict(color = 'forestgreen', width = 1.5), size = 7),
                    hovertemplate = beat_hover),
                row = 2, col = 1)

    else:
        fig = make_subplots(rows = 2, cols = 1,
                            shared_xaxes = True,
                            vertical_spacing = 0.02,
                            row_heights = [0.6, 0.4])

        # ECG/PPG subplot
        fig.add_trace(
            go.Scatter(
                x = x_array,
                y = signal_segment[y],
                name = signal_type,
                showlegend = True,
                line = dict(color = '#3562bd', width = 1.5),
                hovertemplate = f'<b>{signal_type}:</b> %{{y:.2f}} {y_axis} '
                                f'<extra></extra>'),
            row = 1, col = 1)
        fig.update_yaxes(
            title_text = y_axis,
            row = 1, col = 1,
            title_standoff = 5,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey',
            griddash = 'dot',
            tickcolor = 'grey',
            linecolor = 'grey')

        # IBI subplot
        fig.add_trace(
            go.Scatter(
                x = ibi_segment[x],
                y = ibi_segment['IBI'],
                name = 'IBI',
                line = dict(color = '#eb4034', width = 1.5),
                hovertemplate = '<b>IBI</b>: %{y:.2f} ms <extra></extra>'),
            row = 2, col = 1)
        fig.update_yaxes(
            title_text = 'ms',
            row = 2, col = 1, title_standoff = 1,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey',
            griddash = 'dot',
            tickcolor = 'grey',
            linecolor = 'grey')

        # Detected beats
        fig.add_trace(
            go.Scatter(
                x = signal_segment.loc[signal_segment.Beat == 1, x],
                y = signal_segment.loc[signal_segment.Beat == 1, y],
                name = 'Detected Beat',
                showlegend = True,
                mode = 'markers',
                marker = dict(color = '#f9c669', size = 6),
                hovertemplate = beat_hover),
            row = 1, col = 1)

        # Artifactual beats
        fig.add_trace(
            go.Scatter(
                x = signal_segment.loc[signal_segment.Artifact == 1, x],
                y = signal_segment.loc[signal_segment.Artifact == 1, y],
                name = 'Potential Artifact',
                showlegend = True,
                mode = 'markers',
                marker = dict(color = 'red'),
                hovertemplate = artifact_hover),
            row = 1, col = 1)

    # Format shared x-axis
    x_min = signal_segment[x].min()
    x_max = signal_segment[x].max()
    fig.update_xaxes(
        tickfont = dict(size = 14),
        tickcolor = 'grey',
        linecolor = 'grey',
        range = [x_min, x_max])

    # Format figure
    fig.update_layout(
        height = 450,
        title_text = title,
        template = 'simple_white',
        font = dict(family = 'Poppins', color = 'black'),
        legend = dict(
            font = dict(size = 16),
            orientation = 'h',
            yanchor = 'bottom',
            y = 1.05,
            xanchor = 'right',
            x = 1.0),
        annotations = [dict(
            text = x.capitalize(),
            x = 0.5,
            y = -0.22,
            showarrow = False,
            xref = 'paper',
            yref = 'paper',
            font = dict(size = 16)
        )],
        margin = dict(l = 20, r = 20, t = 60, b = 70)
    )
    return fig

def plot_signal(df, x, y, fs, seg_size = 60, segment = 1, n_segments = 1,
                signal_type = None, peaks = None):
    """
    Visualize a signal.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the signal data.
    x : str
        The column containing the x-axis value (e.g., `'Time'`).
    y : str, list
        The column(s) of the signal data (y-axis values).
    fs : int, float
        The sampling rate.
    seg_size : int
        The size of the segment, in seconds; by default, 60.
    segment : int, float, None
        The segment number; by default, 1. For example, segment `1`
        denotes the first segment of the recording. This argument can also
        be set to `None` if `df` contains a 'Segment' column.
    n_segments : int, float
        The number of segments to be visualized; by default, 1.
    signal_type : str
        The type of signal being plotted (i.e., 'ecg', 'bvp', 'acc',
        'ibi'); by default, None.
    peaks : str
        The column containing peak occurrences, i.e., a sequence of
        `0` and/or `1` denoting False or True occurrences of peaks.
        By default, peaks will be plotted on the first trace.

    Returns
    -------
    fig : matplotlib.axes.AxesSubplot
        The signal visualization.
    """

    if segment is None and \
            'segment' in [c.lower() for c in df.columns.tolist()]:
        seg = df.loc[(df.Segment >= 1) & (df.Segment <= 2)]
    else:
        start = int(segment - 1) * seg_size * fs
        end = int(((segment - 1) + n_segments) * seg_size * fs)
        seg = df.iloc[start:end]

    # Set plotting parameters
    plt.rcParams['font.size'] = 14
    palette1 = {'blue': '#4c73c2',
                'red': '#eb4034',
                'green': '#63b068',
                'grey': '#bdbdbd'}
    palette2 = ['#ec2049', '#176196', '#f7db4f', '#63b068']

    # Set up the figure
    fig = go.Figure()

    # Plot a single signal
    if not isinstance(y, list):
        fig.add_trace(go.Scatter(
            x = seg[x],
            y = seg[y],
            mode = 'lines',
            hovertemplate = '%{x}' + '<br>%{y:.2f}' + '<extra></extra>',
            name = f'{y}'))

        # Add peaks
        if peaks != None:
            fig.add_trace(go.Scatter(
                x = seg[x],
                y = np.where(seg[peaks] == 1, seg[y], np.nan),
                mode = 'markers',
                marker = dict(size = 8, color = 'gold', line_width = 1),
                hovertemplate = '<b>Peak</b>: %{y} <extra></extra>',
                name = 'Peaks'))
        fig.update_layout(yaxis_title = y)

    # Plot multiple signals
    else:
        for yval in range(len(y)):
            fig.add_trace(go.Scatter(
                x = seg[x],
                y = seg[y[yval]],
                mode = 'lines',
                line = dict(color = palette2[yval]),
                hovertemplate = '%{x}' + '<br>%{y:.2f}' + '<extra></extra>',
                name = f'{y[yval]}'))

        # Add peaks
        if peaks is not None:
            fig.add_trace(go.Scatter(
                x = seg[x],
                y = np.where(seg[peaks] == 1, seg[y[0]], np.nan),
                mode = 'markers',
                marker = dict(size = 8, color = 'gold', line_width = 1),
                hovertemplate = '<b>Peak</b>: %{y} <extra></extra>',
                name = 'Peaks'))

    # Format the plot
    fig.update_layout(
        xaxis_title = x,
        template = 'simple_white',
        height = 300,
        margin = dict(l = 10, r = 30, b = 50, t = 50, pad = 3)
    )
    # Label axes and set trace colors according to signal type
    if signal_type == 'ecg' or signal_type == 'bvp':
        if isinstance(y, list):
            for d in range(len(fig.data)):
                fig.data[d].line.color = palette2[d]
            return fig.update_layout(yaxis_title = signal_type.upper())
        else:
            return fig.update_traces(
                line_color = palette1['blue']).update_layout(yaxis_title = y)
    elif signal_type == 'acc':
        if isinstance(y, list):
            for d in range(len(fig.data)):
                fig.data[d].line.color = palette2[d]
            return fig.update_layout(yaxis_title = 'm/s<sup>2</sup>')
        else:
            return fig.update_traces(
                line_color = palette1['green']).update_layout(
                yaxis_title = 'm/s<sup>2</sup>')
    elif signal_type == 'ibi':
        if isinstance(y, list):
            for d in range(len(fig.data)):
                fig.data[d].line.color = palette2[d]
            return fig.update_layout(yaxis_title = 'ms')
        else:
            return fig.update_traces(
                line_color = palette1['red']).update_layout(yaxis_title = 'ms')
    else:
        return fig

def plot_ibi_from_ecg(df, x, y, segment, n_segments):
    """
    Visualize an IBI series generated from ECG data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the signal data.
    x : str
        The column containing the x-axis value (e.g., `'Time'`).
    y : str
        The column containing the IBI series (e.g., `'IBI'`).
    segment : int, float
        The segment number. For example, segment `1` denotes the first
        segment of the recording.
    n_segments : int, float
        The number of segments to be visualized; by default, 1.

    Returns
    -------
    fig : matplotlib.axes.AxesSubplot
        The IBI series plot.
    """
    start = int(segment)
    end = round(segment + n_segments)
    seg = df.loc[df['Segment'].between(start, end, inclusive = 'both')]
    plt.rcParams['font.size'] = 14

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = seg[x],
        y = seg[y],
        mode = 'lines',
        marker = dict(color = '#eb4034'),
        hovertemplate = '%{x}' + '<br>%{y:.2f} ms' + '<extra></extra>',
        name = f'{y}'))

    ymin = np.nanmin(seg[y].values.flatten()) * 0.95
    ymax = np.nanmax(seg[y].values.flatten()) * 1.05
    fig.update_layout(
        yaxis_range = (ymin, ymax),
        yaxis_title = 'IBI (ms)',
        xaxis_title = x,
        template = 'simple_white',
        height = 300,
        margin = dict(l = 50, r = 10, b = 50, t = 30, pad = 10)
    )
    fig.update_yaxes(
        title_standoff = 10)
    return fig

def write_beat_editor_file(data, fs, signal_col, beats_col, ts_col = None,
                           filename = None):
    """
    Create a JSON file for input to the Beat Editor.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the cardiac data. Must contain at least
        the following columns:
          - Cardiac signal
          - Beat occurrences labeled as 1
        Optionally, `data` can include:
          - A timestamp column (specified by `ts_col`). If not provided,
            sample indices are used.
          - An "Artifact" column, where artifact occurrences are labeled as 1.
            This allows the Beat Editor to visualize artifactual beat
            locations.
    fs : int
        The sampling frequency of the signal.
    signal_col : str
        The name of the column in `data` containing the cardiac signal.
    beats_col : str
        The name of the column in `data` containing beat occurrences.
    ts_col : str, optional
        The name of the column in `data` containing the timestamps. If not
        provided, timestamps are assumed to correspond to the DataFrame index.
    filename : str, optional
        The name of the JSON file to write. If no filename is provided,
        the default filename 'heartview_edit.json' is used.

    Returns
    -------
    None
    """
    from pathlib import Path

    # Set the output JSON filename
    if filename is None:
        json_filename = 'heartview_edit.json'
    else:
        json_filename = filename + '_edit.json'

    # Check required columns
    required_cols = [('signal_col', signal_col), ('beats_col', beats_col)]
    if ts_col:
        required_cols.append(('ts_col', ts_col))
    for name, col in required_cols:
        if col not in data.columns:
            raise ValueError(f'`{name}` not found in input data.')

    # Check if there are any beats
    if data[beats_col].sum() == 0:
        warnings.warn('No beat occurrences found in input data.', UserWarning)

    # Convert to `datetime` format if provided
    if ts_col is not None:
        data[ts_col] = pd.to_datetime(data[ts_col])
        data.rename(columns = {ts_col: 'Timestamp'}, inplace = True)
    else:
        data.insert(0, 'Sample', data.index + 1)

    # Format columns for JSON keys and save as JSON
    if 'Segment' not in data.columns:
        data.insert(0, 'Segment', (data.index // (fs * 60)) + 1)
    data.rename(columns = {signal_col: 'Signal', beats_col: 'Beat'},
                inplace = True)
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'beat-editor' / 'data'
    data_dir.mkdir(parents = True, exist_ok = True)
    json_path = data_dir / json_filename
    data.to_json(json_path, orient = 'records', date_format = 'epoch',
                 lines = False)
    print(f'Beat Editor JSON file written to {json_path}')

def process_beat_edits(orig_data, edits):
    """
    Apply manual corrections from the Beat Editor output to original data.
    Edits are aligned either by sample index or timestamp, depending on the
    structure of `orig_data`.

    Parameters
    ----------
    orig_data : pd.DataFrame
        A DataFrame containing the original cardiac data inputted to the Beat
        Editor. Must contain a 'Beat' column and either:
        - 'Timestamp' column (datetime), or
        - 'Sample' column (integer sample indices)
    edits : pd.DataFrame
        A DataFrame of edit instructions parsed from a Beat Editor
        `_edited.json` file. Must contain the following columns:
        - 'x': the location of each edit, in the same unit as either
          `orig_data['Timestamp']` (datetime) or `orig_data['Sample']`
          (integer)
        - 'editType': type of edit, with values 'ADD' or 'DELETE'

    Returns
    -------
    processed : pd.DataFrame
        A copy of `orig_data` with the following additional columns:
        - 'Deletion': 1 where beats were deleted, otherwise NaN
        - 'Addition': 1 where beats were added, otherwise NaN
        - 'Unusable': 1 where segments are marked unusable, otherwise NaN
        - 'Edited': 1 where all final beats are, otherwise NaN

    """
    if all(col not in edits.columns for col in ['x', 'from', 'to']):
        raise ValueError('Input edits missing necessary columns.')
    else:
        processed = orig_data.copy()

        # Handle beat insertions/deletions
        if 'x' in edits.columns:
            mask_x = edits['x'].notna()
            if 'Timestamp' in processed.columns:
                edits.loc[mask_x, 'x'] = pd.to_datetime(
                    edits.loc[mask_x, 'x'], unit = 'ms', errors = 'coerce')
            edits.loc[mask_x, 'Sample'] = edits.loc[mask_x, 'x'].apply(
                lambda x: (processed['Timestamp'] - x).abs().idxmin()
                if 'Timestamp' in processed.columns else
                (processed['Sample'] - x).abs().idxmin()
            )

        # Handle unusable markings
        if all(col in edits.columns for col in ['from', 'to']):
            mask_from = edits['from'].notna()
            mask_to = edits['to'].notna()
            if 'Timestamp' in processed.columns:
                edits.loc[mask_from, 'from'] = pd.to_datetime(
                    edits.loc[mask_from, 'from'], unit = 'ms',
                    errors = 'coerce')
                edits.loc[mask_to, 'to'] = pd.to_datetime(
                    edits.loc[mask_to, 'to'], unit = 'ms', errors = 'coerce')

            # Store aligned sample indices
            edits.loc[mask_from, 'Sample_from'] = edits.loc[
                mask_from, 'from'].apply(
                lambda x: (processed['Timestamp'] - x).abs().idxmin()
                if 'Timestamp' in processed.columns else
                (processed['Sample'] - x).abs().idxmin()
            )
            edits.loc[mask_to, 'Sample_to'] = edits.loc[mask_to, 'to'].apply(
                lambda x: (processed['Timestamp'] - x).abs().idxmin()
                if 'Timestamp' in processed.columns else
                (processed['Sample'] - x).abs().idxmin()
            )

        # Identify rows for beat deletion and addition
        deletions = edits.loc[edits.editType == 'DELETE', 'Sample'].values
        additions = edits.loc[edits.editType == 'ADD', 'Sample'].values

        # Flag deletions and additions in the processed DataFrame
        processed.loc[deletions, 'Deletion'] = 1
        processed.loc[additions, 'Addition'] = 1

        # Add 'unusable' labels
        for start, end in zip(edits.Sample_from, edits.Sample_to):
            if pd.notna(start) and pd.notna(end):
                processed.loc[int(start):int(end), 'Unusable'] = 1

        # Add the corrected beat column
        processed['Edited'] = processed.Beat.copy()
        processed.loc[processed.Unusable == 1, 'Edited'] = np.nan
        processed.loc[processed.Deletion == 1, 'Edited'] = np.nan
        processed.loc[processed.Addition == 1, 'Edited'] = 1
        return processed