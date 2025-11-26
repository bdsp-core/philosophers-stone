import pandas as pd
import numpy as np
import os
import sys
import mne
from phi_utils.eeg_fct import filter_routine, spectrogram, eeg_epoch_states
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from ssqueezepy import ssq_cwt, stft, Wavelet


def preprocess_filter(signal, Fs=200, bandpass_low=0.1, bandpass_high=60):
    """filter signals"""

    for col in signal.columns:
        
        if col in ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1', 'cz-oz', 'c3-m1', 'e1-m2', 'e2-m1', 'chin1-chin2', 'chin', 'eeg']:
            # print("highpass set to 20 from 60 for now")
            signal[col] = filter_routine(signal[col].values, Fs, bandpass_low=bandpass_low, bandpass_high=bandpass_high)

        elif col in ['ecg']:
            signal[col] = filter_routine(signal[col].values, Fs, bandpass_low=0.3, bandpass_high=None)
            
        elif col in ['abd', 'chest', 'airflow', 'ptaf', 'resp_airflow']:
            signal[col] = filter_routine(signal[col].values, Fs, bandpass_low=None, bandpass_high=1.7)
            
        elif col in ['lat', 'rat', 'leg']:
            signal[col] = filter_routine(signal[col].values, Fs, bandpass_low=None, bandpass_high=None)
        elif col in ['spo2']:
            # no filter
            continue
        else:
            print(f"no filtering for {col}")

    return signal


### SPECTROGRAM COMPUTING

def compute_eeg_spectrogram(signal_vals, Fs=200, fmax_eeg=50, pad_start=True, hours_pad=11, return_specs_wopad=False,
                            epoch_step_time=1, epoch_time=2, expected_length=None):
    """compute spectrogram of EEG signals.
    signal_vals: 1d numpy array of signals.
    Fs: sampling frequency
    pad_start: pad start with stepsize//2 to have spectrogram result centered at original time series signal.
    hours_pad: if the spectrogram shall be zeropadded in the end for a given amount of hours (to standardize across different recordings).
    epoch_step_time: window step size in seconds.
    epoch_time: window time in seconds.
    """

    specs, freq, eeg_segs = spectrogram(signal_vals[np.newaxis, :], Fs, 
                                        signaltype='eeg', bandwidth=2, epoch_time=epoch_time, 
                                        epoch_step_time=epoch_step_time, fmax=fmax_eeg)
    epoch_state = eeg_epoch_states(eeg_segs, Fs)
    specs = specs[:, 0, :]


    if pad_start:
        specs = np.pad(specs, ((epoch_time//2, epoch_time//2), (0, 0)), 'constant')
        # specs is 1/2 stepsize too long, so resize to original length - epoch_step_time:
        # specs = resize(specs, (specs.shape[0] - epoch_step_time, specs.shape[1]))
        # print('post padding', specs.shape)
        
    if expected_length is not None:
        # first assert that the expected length and the actual length differ by at most 1 minute
        # this is not a hard rule, it depends on the epoch_step_time and epoch configuration
        # it's entirely possible that the length of the spectrogram is not just a simple len(signals)/fs_time divisor.
        # solution: zero-pad.
        max_allowance = 60 * 1 / epoch_step_time
        assert abs(expected_length - specs.shape[0]) <= max_allowance, f"expected length {expected_length} and actual length {specs.shape[0]} differ by more than {max_allowance} samples"
        if expected_length > specs.shape[0]:
            # pad left and right side equally:
            zero_pad = expected_length - specs.shape[0]
            specs = np.pad(specs, ((zero_pad//2, zero_pad//2), (0,0)), 'constant')
            if zero_pad % 2 == 1:
                specs = np.pad(specs, ((1,0), (0,0)), 'constant')
                
        elif expected_length < specs.shape[0]:
            specs = specs[:expected_length, :]
            
        assert specs.shape[0] == expected_length, f"expected length {expected_length} and actual length {specs.shape[0]} differ after 'expected_length' operations."

    if return_specs_wopad:
        specs_no_pad = specs.copy()
        
    spec_len = int(1/epoch_step_time * hours_pad * 3600)  # standardize to 11 hours
    if specs.shape[0] > spec_len:
        specs = specs[:spec_len, :]
    else:
        zero_pad = spec_len - specs.shape[0]
        specs = np.pad(specs, ((0, zero_pad),(0,0)), 'constant')
    assert specs.shape[0] == spec_len
    
    if return_specs_wopad:
        return specs, freq, specs_no_pad

    return specs, freq



### SPECTROGRAM PLOT FUNCTIONS


def plot_spectrogram(specs, freq=None, dt=1, vmin=0.01, vmax=0.99, 
                     spec_db_vmin=None, spec_db_vmax=None, title=None, colorbar_unit='dB',
                     max_freq_to_plot=None):
    """ 
    Plot spectrograms.
    spec: spectrogram
    freq: frequency array
    dt: timeresolution=stepsize of spectrogram. typically 1 second...
    """
    
    
    if freq is None:
        from phi_utils.philosopher_utils import _make_frequency_grid
        try:
            freq = _make_frequency_grid(specs.shape[1])
        except Exception as e:
            print(f"Error in frequency grid creation: {e}")
            print("Please provide a frequency array.")
            raise e
                
    if max_freq_to_plot is not None:
        # only plot frequencies below max_freq_to_plot
        specs = specs[:, freq <= max_freq_to_plot]
        freq = freq[freq <= max_freq_to_plot]
        
    h = int(dt*3600)
    specs[specs==np.inf] = np.nan
    specs[specs==-np.inf] = np.nan
    if spec_db_vmin is None:
        spec_db_vmin = np.nanquantile(specs[h : h * 5, :40].flatten(), vmin)
        if np.isnan(spec_db_vmin):
            spec_db_vmin = np.nanquantile(specs.flatten(), vmin)
    if spec_db_vmax is None:
        spec_db_vmax = np.nanquantile(specs[h : h * 5, :40].flatten(), vmax)
        if np.isnan(spec_db_vmax):
            spec_db_vmax = np.nanquantile(specs.flatten(), vmax)

    fig, ax = plt.subplots(1, 1, figsize=(13, 4), sharex=True)
    divider = make_axes_locatable(ax)

    im = ax.imshow(specs.T, cmap='turbo', origin='lower', aspect='auto',
            extent=(0, specs.shape[0] * dt / 3600, 0, len(freq)),
            vmin=spec_db_vmin, vmax=spec_db_vmax)

    cax = fig.add_axes([0.96, 0.45, 0.01, 0.4])
    cax.tick_params(length=2, labelsize=7, pad=1)
    if colorbar_unit == 's^2/Hz':
        plt.colorbar(im, cax=cax, aspect=100, ticks=[np.ceil(spec_db_vmin*1e2)/1e2, np.round((spec_db_vmin+spec_db_vmax)/2, 2), np.floor(spec_db_vmax*1e2)/1e2])
        cax.text(0, spec_db_vmax*1.1, colorbar_unit)
    elif colorbar_unit == 'Amplitude':
        plt.colorbar(im, cax=cax, aspect=100, ticks=[int(spec_db_vmin), int((spec_db_vmin+spec_db_vmax)/2), int(spec_db_vmax)])
        cax.text(0, spec_db_vmax + 0.5, colorbar_unit)
    else:
        plt.colorbar(im, cax=cax, aspect=100, ticks=[int(spec_db_vmin), int((spec_db_vmin+spec_db_vmax)/2), int(spec_db_vmax-1)])
        cax.text(0, spec_db_vmax + 0.5, colorbar_unit)

    ax.set_xlabel("Time (hours)", labelpad=0)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    # set y ticks. freq is not necessarily linearly spaced.
    yticks = np.linspace(0, len(freq)-1, 10).astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{freq[y]:.0f}" for y in yticks])

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95)
    return fig


def plot_spectrogram_with_stages(specs, stage, freq=None, dt=1, vmin=0.01, vmax=0.99, 
                                 spec_db_vmin=None, spec_db_vmax=None, title=None, 
                                 colorbar_unit='dB', figsize=(13, 6), 
                                 max_freq_to_plot=None, fontsize_labels=8):
    """ 
    Plot spectrograms and stage annotations.
    spec: spectrogram
    freq: frequency array
    dt: timeresolution=stepsize of spectrogram. typically 1 second...
    stage: stage annotations
    """
    
    if freq is None:
        from phi_utils.philosopher_utils import _make_frequency_grid
        try:
            freq = _make_frequency_grid(specs.shape[1])
        except Exception as e:
            print(f"Error in frequency grid creation: {e}")
            print("Please provide a frequency array.")
            raise e
            
    stage = stage.flatten()
    assert specs.shape[0] == len(stage), f"specs.shape[0] {specs.shape[0]} != stage.shape[0] {len(stage)}"
    
    h = int(dt*3600)
    specs[specs==np.inf] = np.nan
    specs[specs==-np.inf] = np.nan
    if spec_db_vmin is None:
        spec_db_vmin = np.nanquantile(specs[h : h * 5, :40], vmin)
        if np.isnan(spec_db_vmin):
            spec_db_vmin = np.nanquantile(specs, vmin)
        
    if spec_db_vmax is None:
        spec_db_vmax = np.nanquantile(specs[h : h * 5, :40], vmax)
        if np.isnan(spec_db_vmax):
            spec_db_vmax = np.nanquantile(specs, vmax)

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [10, 1]})
    divider = make_axes_locatable(ax[0])
    im = ax[0].imshow(specs.T, cmap='turbo', origin='lower', aspect='auto',
            extent=(0, specs.shape[0]/3600, 0, freq.max()),
            vmin=spec_db_vmin, vmax=spec_db_vmax)

    cax = fig.add_axes([0.96, 0.55, 0.01, 0.3])
    cax.tick_params(length=2, labelsize=7, pad=1)

    plt.colorbar(im, cax=cax, aspect=100)

    # Stages / Hypnogram:
    palette_stage = ['darkblue', 'blue', 'lightblue', 'purple', 'gold']
    vals_hypno_tmp = np.zeros((len(stage), 5))
    vals_hypno_tmp[:] = np.nan
    for i, val in enumerate(stage):
        if pd.isna(val) or val < 1 or val > 5:
            continue
        vals_hypno_tmp[i, int(val - 1)] = 1

    index = np.arange(0, len(vals_hypno_tmp)) / 3600 / dt  # time in hours
    vals_hypno_df = pd.DataFrame(vals_hypno_tmp, columns=['N3', 'N2', 'N1', 'R', 'W'], index=index)

    vals_hypno_df.plot(kind='area', color=palette_stage, alpha=.8, 
                        ax=ax[1], sharex=True,
                        stacked=True, lw=0, legend=True)

    ax[1].get_legend().remove()
    ax[1].set_ylim([0, 1])
    ax[1].set_yticks([])

    ### End stages
    
    # Add legend above, horizontally
    handles, labels = ax[1].get_legend_handles_labels()
    ax[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.75, 1.7),
                 fontsize=9, frameon=False, labelspacing=0, ncol=5, columnspacing=0.5)
    
    
    ax[-1].set_xlabel("Time (hours)", labelpad=0)
    ax[0].set_ylabel("Frequency (Hz)")
    ax[1].set_ylabel("Stage", labelpad=2)
    ax[0].set_title(title)
    
    ax[-1].tick_params(axis='both', length=2, labelsize=fontsize_labels)
    ax[0].tick_params(axis='both', length=2, labelsize=fontsize_labels)


    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95)
    
    return fig


def plot_spectrogram_signals(specs, freq, signal, fs, dt=1, annotations=None, vmin=0.01, vmax=0.99, spec_db_vmin=None, spec_db_vmax=None, hours_pad=11, title=None, colorbar_unit='dB', figsize=(13, 6)):
    """ 
    Plot spectrograms and time-domain signal.
    spec: spectrogram
    freq: frequency array
    dt: timeresolution=stepsize of spectrogram. typically 1 second...
    signal: timedomain signal
    fs: sampling frequency of timedomain signal
    annotations: vectorized annotations array, either stage, resp, arousal or limb.
    """

    h = int(dt*3600)
    specs[specs==np.inf] = np.nan
    specs[specs==-np.inf] = np.nan
    if spec_db_vmin is None:
        spec_db_vmin = np.nanquantile(specs[h : h * 5, :40], vmin)
        if np.isnan(spec_db_vmin):
            spec_db_vmin = np.nanquantile(specs, vmin)
        
    if spec_db_vmax is None:
        spec_db_vmax = np.nanquantile(specs[h : h * 5, :40], vmax)
        if np.isnan(spec_db_vmax):
            spec_db_vmax = np.nanquantile(specs, vmax)

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    divider = make_axes_locatable(ax[0])
    im = ax[0].imshow(specs.T, cmap='turbo', origin='lower', aspect='auto',
            extent=(0, specs.shape[0]/3600, 0, freq.max()),
            vmin=spec_db_vmin, vmax=spec_db_vmax)

    cax = fig.add_axes([0.96, 0.55, 0.01, 0.3])
    cax.tick_params(length=2, labelsize=7, pad=1)

    if colorbar_unit == 's^2/Hz':
        plt.colorbar(im, cax=cax, aspect=100, ticks=[np.ceil(spec_db_vmin*1e2)/1e2, np.round((spec_db_vmin+spec_db_vmax)/2, 2), np.floor(spec_db_vmax*1e2)/1e2])
        cax.text(0, spec_db_vmax*1.1, colorbar_unit)
    else:
        plt.colorbar(im, cax=cax, aspect=100, ticks=[int(spec_db_vmin), int((spec_db_vmin+spec_db_vmax)/2), int(spec_db_vmax-1)])
        cax.text(0, spec_db_vmax + 0.5, colorbar_unit)

    if hours_pad > 0:
        len_pad_hours = fs * 3600 * 11
        zero_pad = len_pad_hours - len(signal)
        if zero_pad > 0:
            signal = np.pad(signal, (0, (zero_pad)), 'constant')
            if annotations is not None:
                annotations = np.pad(annotations.flatten(), (0, (zero_pad)), 'constant')

    ax[1].plot(np.arange(0, len(signal))/fs/3600, signal, lw=0.5)
    if annotations is not None:
        annotations = annotations.flatten()
        ax[1].plot(np.arange(0, len(annotations))/fs/3600, annotations * 20)

    ax[1].set_ylim([-200, 200])
    ax[1].set_xlabel("Time (hours)", labelpad=0)
    ax[0].set_ylabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude", labelpad=0)
    ax[0].set_title(title)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95)
    
    return fig




# WAVELETS:



# Part A: Signal Definition Module

def define_signal(signal_type, N, fs):
    t = np.arange(N) / fs
    if signal_type == 'sum_sinusoids':
        x = 1 * np.cos(2 * np.pi * 10 * t) + 1/2 * np.cos(2 * np.pi * 25 * t)
    elif signal_type == 'self_reflected_noise':
        xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
        xo += xo[::-1]  # add self-reflected
        x = xo + np.sqrt(2) * np.random.randn(N)  # add noise
    elif signal_type == 'chirp':
        x = np.cos(2 * np.pi * 10 * t + 10 * t**2)
    else:
        raise ValueError("Unsupported signal type")
    return x, t


# Part B: Time-Frequency Representation Module

def compute_wavelet_transform(x, wavelet='gmw', nv=32, fs=200):
    # Compute the wavelet transform

    # len_x_threshold = 3600000 # 10 hours with fs=100
    len_x_threshold = 2520000 # this applies to most of the EEG signals for a full night. (7 hours with fs=100)

    if len(x) > len_x_threshold:
        vectorized = False
    else: 
        vectorized = True
    # Note: for a typical 1D EEG full night signal (length somewhere around 250k-300k),
    # vectorized=True: 40 sec, vectorized=False: 150 sec.
    
    Twx, Wx, ssq_freqs, scales, *_ = ssq_cwt(x, wavelet, fs=fs, nv=nv, vectorized=vectorized)
    
    Wx = np.abs(Wx)[::-1]  # flip to make origin lower
    Twx = np.abs(Twx)[::-1]  # flip to make origin lower
    ssq_freqs = ssq_freqs[::-1]

    Wx = Wx.T  # Transpose to match the shape of the STFT. Now, Wx.shape = (N, len(ssq_freqs))
    Twx = Twx.T

    return Twx, Wx, ssq_freqs

def compute_tf_representations(x, fs, wavelet='gmw', nv=32):
    # STFT computation
    Sx = stft(x)[::-1]
    freqs_stft = np.linspace(1, 0, len(Sx)) * fs/2
    # Process data for visualization
    Sx = np.abs(Sx)[::-1]  # flip to make origin lower
    freqs_stft = freqs_stft[::-1]

    # CWT and SSQ CWT computation
    Twx, Wx, ssq_freqs = compute_wavelet_transform(x, wavelet=wavelet, nv=nv, fs=fs)

    return Sx, freqs_stft, Wx, Twx, ssq_freqs


def pad_spectrogram(specs, fs_time, hours_pad=11):
    """
    Pad the spectrogram to a fixed length of 11 hours.
    specs: np.array, shape (time, freq)
    fs_time: float, time resolution in Hz
    hours_pad: int, number of hours to pad the spectrogram to
    """
    
    spec_len_desired = int(fs_time * hours_pad * 3600)  # standardize to 11 hours
    if specs.shape[0] > spec_len_desired:
        specs = specs[:spec_len_desired, :]
    else:
        zero_pad = spec_len_desired - specs.shape[0]
        specs = np.pad(specs, ((0, zero_pad),(0,0)), 'constant')
    assert specs.shape[0] == spec_len_desired, f"specs.shape[0] not equal to spec_len_desired: {specs.shape[0]} != {spec_len_desired}"
    
    return specs

    
def interpolate_wx_2d(Wx, old_freqs, new_freqs, fs_old, fs_new):
    """
    Interpolates the wavelet transform matrix Wx over both frequency and time dimensions.

    Parameters:
        Wx (numpy.ndarray): The wavelet transform matrix of shape (time, freqs).
        old_freqs (numpy.ndarray): Original frequency array associated with Wx.
        new_freqs (numpy.ndarray): New frequency array to which Wx is interpolated.
        fs_old (int): Original time-resolution sampling frequency.
        fs_new(int): New time-resolution sampling frequency.

    Returns:
        numpy.ndarray: Interpolated wavelet transform matrix over both dimensions.
    """
    
    old_time_points = Wx.shape[0]
    new_time_points = int(old_time_points * fs_new / fs_old)

    # Create the original time array
    old_times = np.linspace(0, old_time_points / fs_old, num=old_time_points, endpoint=False)
    new_times = np.linspace(0, old_time_points / fs_old, num=new_time_points, endpoint=False)

    # Interpolate in the time domain for each frequency
    Wx_time_interpolated = np.zeros((new_time_points, len(old_freqs)))
    for f in range(len(old_freqs)):
        time_interp = interp1d(old_times, Wx[:, f], kind='linear', bounds_error=False, fill_value="extrapolate")
        Wx_time_interpolated[:, f] = time_interp(new_times)

    # Interpolate in the frequency domain for each new time point
    Wx_interpolated = np.zeros((new_time_points, len(new_freqs)))
    for t in range(new_time_points):
        freq_interp = interp1d(old_freqs, Wx_time_interpolated[t, :], kind='linear', bounds_error=False, fill_value="extrapolate")
        Wx_interpolated[t, :] = freq_interp(new_freqs)

    return Wx_interpolated
