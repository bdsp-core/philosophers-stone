import pandas as pd
import numpy as np
import os
import sys
import mne
from load_edf import load_edf_eeg, load_edf_respeffort, load_edf_ecg
from eeg_fct import eeg_filter, spectrogram, eeg_epoch_states, wx_interpolation, wavelet_frequency_selection
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator

from ssqueezepy import ssq_cwt, Wavelet
from preprocessing_and_spectrograms import *
from load_data import load_prepared_data

n_cpu = os.cpu_count()


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

from ssqueezepy import ssq_cwt, stft

def compute_wavelet_transform(x, wavelet='gmw', nv=32, fs=200):
    # Compute the wavelet transform

    if len(x) > 3600000: # 10 hours with fs=100
        vectorized = False
    else: 
        vectorized = True
        
    vectorized = True
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
    print('ssq_cwt')
    Twx, Wx, ssq_freqs = compute_wavelet_transform(x, wavelet=wavelet, nv=nv, fs=fs)
    print('ssq_cwt done')

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


import matplotlib.pyplot as plt

figsize = (4, 4)

def plot_spectrogram2(Sx, freqs_stft, fs, title='Spectrogram', xlabel='Time', ylabel='Frequency'):
    plt.figure(figsize=figsize)
    plt.imshow(Sx, cmap='turbo', aspect='auto', origin='lower',
                extent=(0, Sx.shape[1] / fs, freqs_stft.min(), freqs_stft.max()))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_cwt(Wx, ssq_freqs, title='CWT'):
    show_every_nth = 10
    yticks = np.arange(0, len(ssq_freqs), show_every_nth)
    yticklabels = np.round(ssq_freqs[yticks], 2)
    plt.figure(figsize=figsize)
    plt.imshow(Wx, cmap='turbo', aspect='auto', origin='lower')
    plt.yticks(yticks, labels=yticklabels)
    plt.title(title)
    plt.show()

def plot_ssqt(Twx, ssq_freqs, title='SSQ CWT'):
    show_every_nth = 10
    yticks = np.arange(0, len(ssq_freqs), show_every_nth)
    yticklabels = np.round(ssq_freqs[yticks], 2)
    plt.figure(figsize=figsize)
    plt.imshow(Twx, cmap='turbo', aspect='auto', origin='lower')
    plt.yticks(yticks, labels=yticklabels)
    plt.title(title)
    plt.show()

def plot_multitaper(specs_mltp, freqs_mltp, duration, fs, title='Multitaper Spectrogram'):
    plt.figure(figsize=figsize)
    plt.imshow(specs_mltp, cmap='turbo', origin='lower', aspect='auto',
                extent=(0, duration, freqs_mltp.min(), freqs_mltp.max()))
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Magnitude')
    plt.show()

def limit_frequencies(Sx, freqs_stft, Wx, Twx, ssq_freqs, f_low, f_high):
    # Limit STFT
    valid_idxs_stft = (freqs_stft >= f_low) & (freqs_stft <= f_high)
    Sx = Sx[valid_idxs_stft]
    freqs_stft = freqs_stft[valid_idxs_stft]

    # Limit CWT and SSQ CWT
    valid_idxs_ssqt = (ssq_freqs >= f_low) & (ssq_freqs <= f_high)
    Wx = Wx[valid_idxs_ssqt]
    Twx = Twx[valid_idxs_ssqt]
    ssq_freqs = ssq_freqs[valid_idxs_ssqt]

    return Sx, freqs_stft, Wx, Twx, ssq_freqs


def convert_scalogram_to_linear_scale(Wx, ssq_freqs, fs):
    # Define the new linear frequency scale
    linear_freqs = np.linspace(ssq_freqs.min(), ssq_freqs.max(), len(ssq_freqs))
    
    # Interpolate the wavelet transform data to the new frequency scale
    # We need to interpolate each time slice (each column of Wx) separately
    Wx_linear = np.zeros_like(Wx)
    
    for i in range(Wx.shape[1]):  # Iterate over each time point
        # Create an interpolation function for the current time slice
        interpolator = interp1d(ssq_freqs, Wx[:, i], bounds_error=False, fill_value="extrapolate")
        
        # Use the interpolator to compute the values at the new linear frequency points
        Wx_linear[:, i] = interpolator(linear_freqs)
    
    return Wx_linear, linear_freqs


def plot_all_representations(Sx, freqs_stft, Wx, ssq_freqs, Twx, specs_mltp, freqs_mltp, duration, fs, N):
    plt.figure(figsize=(12, 3))

    # Plot Multitaper Spectrogram first
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(specs_mltp, cmap='turbo', origin='lower', aspect='auto',
                extent=(0, duration, freqs_mltp.min(), freqs_mltp.max()))
    ax1.set_title('Multitaper Spectrogram')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')

    # Plot STFT
    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(Sx, cmap='turbo', aspect='auto', origin='lower',
                extent=(0, duration, freqs_stft.min(), freqs_stft.max()))
    ax2.set_title('STFT')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')

    # Plot CWT
    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(Wx, cmap='turbo', aspect='auto', origin='lower',
                extent=(0, N, 0, len(ssq_freqs)))  # Extent in sample indices for x-axis
    show_every_nth = 15
    yticks = np.arange(0, len(ssq_freqs), show_every_nth)
    yticklabels = np.round(ssq_freqs[yticks], 2)
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(yticklabels)
    ax3.set_title('CWT')
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Frequency (Hz)')

    # Plot SSQ CWT
    ax4 = plt.subplot(1, 4, 4)
    ax4.imshow(Twx, cmap='turbo', aspect='auto', origin='lower',
                extent=(0, N, 0, len(ssq_freqs)))  # Same extent adjustment as CWT
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(yticklabels)
    ax4.set_title('SSQ CWT')
    ax4.set_xlabel('Samples')
    ax4.set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()



def plot_all_data(signal_data, stage_data, Sx, freqs_stft, Wx, ssq_freqs, Twx, specs_mltp, freqs_mltp, Wx_linear, Twx_linear, linear_freqs,
                    fs, wavelet_name=None, N_wavelet=None, figsize=None, save_fig=False, plot_savedir=None, save_str=None):
    
    if figsize is None:
        figsize = (10, 9)
    fontsize = 7
    if figsize[0] < 10:
        fontsize = 5
        
    fig, axes = plt.subplots(8, 1, sharex=True,
                            gridspec_kw={'height_ratios': [1, 3, 2, 2, 2, 2, 2, 2]}, figsize=figsize)

    time = np.arange(len(signal_data)) / fs
    # Plot sleep staging data
    axes[0].plot(time, stage_data, color='m')
    axes[0].set_ylabel("Sleep\nStages", fontsize=fontsize)
    # axes[0].set_ylabel("Stage")
    
    # Plot original signal data
    axes[1].plot(time, signal_data, linewidth=0.5)
    axes[1].set_ylabel("Original Signal", fontsize=fontsize)
    # axes[1].set_ylabel("Amplitude")

    # Plot Multitaper Spectrogram
    vmin = np.percentile(specs_mltp, 5)
    vmax = np.percentile(specs_mltp, 95)
    axes[2].imshow(specs_mltp, cmap='turbo', aspect='auto', origin='lower', extent=(0, len(signal_data)/fs, freqs_mltp.min(), freqs_mltp.max()), vmin=vmin, vmax=vmax)
    axes[2].set_ylabel("Multitaper Spectrogram", fontsize=fontsize)
    # axes[2].set_ylabel("Frequency (Hz)")

    # Plot STFT
    vmin = np.percentile(Sx, 5)
    vmax = np.percentile(Sx, 95)
    axes[3].imshow(Sx, cmap='turbo', aspect='auto', origin='lower', extent=(0, len(signal_data)/fs, freqs_stft.min(), freqs_stft.max()), vmin=vmin, vmax=vmax)
    axes[3].set_ylabel("STFT", fontsize=fontsize)
    # axes[3].set_ylabel("Frequency (Hz)")

    for i_axis in range(4):
        for label in (axes[i_axis].get_xticklabels() + axes[i_axis].get_yticklabels()):
            label.set_fontsize(fontsize - 1)
            
    # Plot CWT
    vmin = np.percentile(Wx, 5)
    vmax = np.percentile(Wx, 90)
    axes[4].imshow(Wx, cmap='turbo', aspect='auto', origin='lower', extent=(0, len(signal_data)/fs, 0, len(ssq_freqs)), vmin=vmin, vmax=vmax)
    idx_yticks = np.linspace(0, len(ssq_freqs) - 1, num=5)
    idx_yticks = idx_yticks.astype(int)
    yticklabels = np.round(ssq_freqs[idx_yticks], 2)
    axes[4].set_yticks(idx_yticks)
    axes[4].set_yticklabels(yticklabels, fontsize=fontsize - 1)
    axes[4].set_ylabel("CWT", fontsize=fontsize)
    # axes[4].set_ylabel("Frequency (Hz)")

    # Plot SSQ CWT
    vmin = np.percentile(Twx, 5)
    vmax = np.percentile(Twx, 95)
    axes[5].imshow(Twx, cmap='turbo', aspect='auto', origin='lower', extent=(0, len(signal_data)/fs, 0, len(ssq_freqs)), vmin=vmin, vmax=vmax)
    axes[5].set_yticks(idx_yticks)
    axes[5].set_yticklabels(yticklabels, fontsize=fontsize - 1)
    axes[5].set_ylabel("SSQ CWT", fontsize=fontsize)
    # axes[5].set_ylabel("Frequency (Hz)")
    axes[5].set_xlabel("Time (s)")
    
    # Plot CWT Linear
    vmin = np.percentile(Wx_linear, 5)
    vmax = np.percentile(Wx_linear, 95)
    axes[6].imshow(Wx_linear, cmap='turbo', aspect='auto', origin='lower', extent=(0, len(signal_data)/fs, 0, len(linear_freqs)), vmin=vmin, vmax=vmax)
    axes[6].set_yticks(np.linspace(0, len(linear_freqs), num=5))
    axes[6].set_yticklabels(np.round(np.linspace(linear_freqs.min(), linear_freqs.max(), num=5), 2), fontsize=fontsize - 1)
    axes[6].set_ylabel("CWT\n(linear)", fontsize=fontsize)
    # axes[4].set_ylabel("Frequency (Hz)")

    # Plot SSQ CWT Linear
    vmin = np.percentile(Twx_linear, 5)
    vmax = np.percentile(Twx_linear, 95)
    axes[7].imshow(Twx_linear, cmap='turbo', aspect='auto', origin='lower', extent=(0, len(signal_data)/fs, 0, len(linear_freqs)), vmin=vmin, vmax=vmax)
    axes[7].set_yticks(np.linspace(0, len(linear_freqs), num=5))
    axes[7].set_yticklabels(np.round(np.linspace(linear_freqs.min(), linear_freqs.max(), num=5), 2), fontsize=fontsize - 1)
    axes[7].set_ylabel("SSQ CWT\n(linear)", fontsize=fontsize)
    # axes[5].set_ylabel("Frequency (Hz)")
    axes[7].set_xlabel("Time (s)")


    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    
    if save_fig:
        assert plot_savedir is not None, "Please provide a directory to save the plot"
        assert save_str is not None, "Please provide a string to save the plot"
        
        plt.savefig(os.path.join(plot_savedir, f"tf_analysis_all_{save_str}.png"), dpi=500)
        
    plt.show()
    
    
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