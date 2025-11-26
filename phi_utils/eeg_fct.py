import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


import numpy as np
from mne.filter import filter_data, notch_filter
from mne.time_frequency import psd_array_multitaper
from scipy.signal import detrend
from collections import Counter
from scipy.interpolate import interp2d

def eeg_filter(eeg, Fs, notch_freq=60., bandpass_low=0.02, bandpass_high=60):
    """
    eeg filter
    """
    
    notch_freq = notch_freq  # [Hz]
    bandpass_freq = [bandpass_low, bandpass_high]  # [Hz]
    
    # filter EEG
    if notch_freq is not None:
        eeg = notch_filter(eeg, Fs, notch_freq, verbose=False)
    if bandpass_freq is not None:
        eeg = filter_data(eeg, Fs, bandpass_freq[0], bandpass_freq[1], verbose=False)

    return eeg


def filter_routine(signal, Fs, notch_freq=60., bandpass_low=0.02, bandpass_high=60):
    """
    filter routine, notch and bandpass
    """
    notch_freq = notch_freq  # [Hz]
    bandpass_freq = [bandpass_low, bandpass_high]  # [Hz]
    
    # filter EEG
    if notch_freq is not None:
        signal = notch_filter(signal, Fs, notch_freq, verbose=False)
    if (bandpass_freq is not None) & ((bandpass_low is not None) | (bandpass_high is not None)):
        signal = filter_data(signal, Fs, bandpass_freq[0], bandpass_freq[1], verbose=False)

    return signal



def spectrogram(signal, Fs, signaltype=None, epoch_time=30, epoch_step_time=30, decibel=True, fmin=0.02, fmax=60, bandwidth=None, adaptive=True, n_jobs=1):
    """
    Inputs:
    signal: 1d numpy array of signal (time domain)
    Fs: sampling frequency
    signaltype: keywords/shortcuts (see code below, selects bandwith based on keyword)
    epoch_time: window-length in seconds
    epoch_step_time: stepsize in seconds
    decibel: boolean, if result shall be return in decibel (default True)
    fmin: minimum frequency of interest
    fmax: maximum frequency of interest
    bandwidth: multi-taper bandwidth parameter
    adaptive: (see MNE description. True=more accurate but slow)
    n_jobs: parallel jobs.
    Returns:
    # specs.shape = (#epoch, #channel, #freq)
    # freq.shape = (#freq,)
    """
    
    if n_jobs == 'max':
        import multiprocessing
        n_jobs = max(multiprocessing.cpu_count() - 1, 1)

    # segment
    epoch_size = int(round(epoch_time*Fs))
    epoch_step = int(round(epoch_step_time*Fs))
    start_ids = np.arange(0, signal.shape[1]-epoch_size+epoch_step, epoch_step)
    seg_ids = list(map(lambda x: np.arange(x, x+epoch_size), start_ids))
    signal_segs = signal[:, seg_ids].transpose(1, 0, 2)  # signal_segs.shape=(#epoch, #channel, Tepoch)
    # compute spectrogram

    if bandwidth is None:
        if signaltype == 'eeg':       
            NW = 10.
            bandwidth = NW*2./epoch_time
        elif signaltype == 'resp_effort':
            NW = 1
            bandwidth = NW/epoch_time
        else:
            raise ValueError("Unexpected signaltype! ")

    # experimenting values with toy data:
    # bandwidth = 1
    # half_nbw = 0.55
    # bandwidth = half_nbw / (epoch_time  * Fs / (2. * Fs))
    # print(bandwidth)

    # this is how half nbw is computed in code:
    n_times = signal_segs.shape[-1]
    half_nbw = float(bandwidth) * n_times / (2. * Fs)
    n_tapers_max = int(2 * half_nbw)
    # print(half_nbw)
    # print(n_tapers_max)
    specs, freq = psd_array_multitaper(signal_segs, Fs, fmin=fmin, fmax=fmax, adaptive=adaptive, low_bias=True, verbose='ERROR', bandwidth=bandwidth, normalization='full', n_jobs=n_jobs);

    specs[np.isnan(specs)] = 0
    
    if decibel:
        specs[specs == 0] = 1e-25 # avoid log(0)
        specs = 10*np.log10(specs)
    
    # if specs has nan, raise error:
    if np.any(np.isnan(specs)):
        raise ValueError('Spectrogram contains NaN')
    
    return specs, freq, signal_segs
    

def eeg_epoch_states(eeg_segs, Fs, verbose=False):

    std_thres = 0.2  # [uV]
    std_thres2 = 1.  # [uV]
    flat_seconds = 5  # [s]
    amplitude_thres = 500  # [uV]

    if flat_seconds > eeg_segs.shape[2] // Fs: 
        flat_seconds = eeg_segs.shape[2] // Fs #flat sec = eeg segment length

    # mark artifact epochs
    flat_length = int(round(flat_seconds*Fs))
    epoch_state = ['good']*eeg_segs.shape[0]

    # mark epochs with NaN
    nan2d = np.any(np.isnan(eeg_segs), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    for i in nan1d:
        epoch_state[i] = 'EEG contains NaN'

    # mark large amplitude
    amplitude_large2d = np.any(np.abs(eeg_segs)>amplitude_thres, axis=2)
    amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0]
    for i in amplitude_large1d:
        epoch_state[i] = 'Large amplitude: higher than %guV'%amplitude_thres

    # mark flat signal with flat_length
    short_segs = eeg_segs.reshape(eeg_segs.shape[0], eeg_segs.shape[1], eeg_segs.shape[2]//flat_length, flat_length)
    flat2d = np.any(detrend(short_segs, axis=3).std(axis=3)<=std_thres, axis=2)
    flat2d = flat2d|(np.std(eeg_segs,axis=2)<=std_thres2)
    flat1d = np.where(np.any(flat2d, axis=1))[0]
    for i in flat1d:
        epoch_state[i] = 'Flat signal'
    if verbose:
        print(Counter(epoch_state))
    
    return epoch_state


def wx_interpolation(wx, ssq_freqs, f_new):
    """
    cubic spline interpolation of wavelet scalogram and map it to desired new frequency range in linear scale (i.e. compute linear "spectrogram").
    """

    f_interp = interp2d(np.arange(wx.shape[1]), ssq_freqs, wx, kind='cubic')        
    wx_interp = f_interp(np.arange(wx.shape[1]), f_new)

    return wx_interp


def wavelet_frequency_selection(wx, twx, ssq_freqs, freq_min, freq_max):
    """
    frequency selection for wavenet output
    """
    wx_sel = np.abs(wx[(freq_min <= ssq_freqs) & (ssq_freqs <= freq_max), :])
    twx_sel = np.abs(twx[(freq_min <= ssq_freqs) & (ssq_freqs <= freq_max), :])
    ssq_freqs_sel = ssq_freqs[(freq_min <= ssq_freqs) & (ssq_freqs <= freq_max)]

    return wx_sel, twx_sel, ssq_freqs_sel
