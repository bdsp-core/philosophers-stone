import datetime
import re
import numpy as np
import mne


def load_edf_eeg(path, shhs=False):
    """
    convencience function, load data from edf.
    shhs: set True if data is part of SHHS.
    """

    edf = mne.io.read_raw_edf(path, stim_channel=None, preload=False, verbose=False)
    edf_channels = edf.info['ch_names']
    Fs = edf.info['sfreq']
    if type(edf.info['meas_date'])==tuple:
        start_time = datetime.datetime.fromtimestamp(edf.info['meas_date'][0])+ timedelta(seconds=time.altzone)
    else:
        start_time = edf.info['meas_date']

    # find EEG channels
    if shhs: # only 2 EEGs are available in SHHS
        eeg_channels = ['EEG', 'EEG(sec)']
        eeg = edf.get_data(picks=eeg_channels)  # eeg.shape=(#channel, T)
        eeg_channels = ['C4-M1', 'C3-M2'] # as by SHHS docs.
        
    else:
        eeg_channels_regex1 = ['^F3-(A|M)', '^F4-(A|M)', '^C3-(A|M)', '^C4-(A|M)', '^O1-(A|M)', '^O2-(A|M)']
        eeg_channels_regex2 = ['^(A|M)1$', '^(A|M)2$', '^C3$', '^C4$', '^O1$', '^O2$', '^F3$', '^F4$']
        all_ok_channels1 = [[x for x in edf_channels if re.match(regex, x)] for regex in eeg_channels_regex1]
        all_ok_channels2 = [[x for x in edf_channels if re.match(regex, x)] for regex in eeg_channels_regex2]
        if all([len(x)==1 for x in all_ok_channels1]):
            eeg_channels = [x[0] for x in all_ok_channels1]
            eeg = edf.get_data(picks=eeg_channels)  # eeg.shape=(#channel, T)
        elif all([len(x)==1 for x in all_ok_channels2]):
            eeg_channels = [x[0] for x in all_ok_channels2]
            eeg = edf.get_data(picks=eeg_channels)
            eeg = np.array([eeg[6]-eeg[1],  # F3-M2
                            eeg[7]-eeg[0],  # F4-M1
                            eeg[2]-eeg[1],  # C3-M2
                            eeg[3]-eeg[0],  # C4-M2
                            eeg[4]-eeg[1],  # O1-M2
                            eeg[5]-eeg[0],])# O2-M2

        else:
            raise ValueError(f'EEG not found. Contained channels: {edf_channels}')

    # convert to uV
    eeg = eeg*1e6
    
    return eeg, Fs, eeg_channels, start_time
    

def load_edf_respeffort(path):
    """
    convencience function, load data from edf.
    shhs: set True if data is part of SHHS.
    """


    edf = mne.io.read_raw_edf(path, stim_channel=None, preload=False, verbose=False)
    edf_channels = edf.info['ch_names']
    Fs = int(edf.info['sfreq'])

    respiratory_effort_channels = [x for x in edf_channels if any([y in x.lower() for y in ['effort', 'chest', 'tho', 'abd']])]
    if len(respiratory_effort_channels) == 0:
        raise ValueError("No breathing channel with keywords ['effort', 'chest', 'tho', 'abd'] found! EDF contains: {edf_channels}")
        
    if type(edf.info['meas_date'])==tuple:
        start_time = datetime.datetime.fromtimestamp(edf.info['meas_date'][0])+ timedelta(seconds=time.altzone)
    else:
        start_time = edf.info['meas_date']

    resp = edf.get_data(picks=respiratory_effort_channels)  # eeg.shape=(#channel, T) 
    
    return resp, Fs, respiratory_effort_channels, start_time
    


def load_edf_ecg(path, shhs=False):
    """
    convencience function, load data from edf.
    shhs: set True if data is part of SHHS.
    """

    edf = mne.io.read_raw_edf(path, stim_channel=None, preload=False, verbose=False)
    edf_channels = edf.info['ch_names']
    Fs = edf.info['sfreq']
    if type(edf.info['meas_date'])==tuple:
        start_time = datetime.datetime.fromtimestamp(edf.info['meas_date'][0])+ timedelta(seconds=time.altzone)
    else:
        start_time = edf.info['meas_date']

    # find EEG channels
    if shhs: # only 2 EEGs are available in SHHS
        eeg_channels = ['EEG', 'EEG(sec)']
        eeg = edf.get_data(picks=eeg_channels)  # eeg.shape=(#channel, T)
        eeg_channels = ['C4-M1', 'C3-M2'] # as by SHHS docs.
        
    else:
        eeg_channels_regex1 = ['^F3-(A|M)', '^F4-(A|M)', '^C3-(A|M)', '^C4-(A|M)', '^O1-(A|M)', '^O2-(A|M)']
        eeg_channels_regex2 = ['^(A|M)1$', '^(A|M)2$', '^C3$', '^C4$', '^O1$', '^O2$', '^F3$', '^F4$']
        all_ok_channels1 = [[x for x in edf_channels if re.match(regex, x)] for regex in eeg_channels_regex1]
        all_ok_channels2 = [[x for x in edf_channels if re.match(regex, x)] for regex in eeg_channels_regex2]
        if all([len(x)==1 for x in all_ok_channels1]):
            eeg_channels = [x[0] for x in all_ok_channels1]
            eeg = edf.get_data(picks=eeg_channels)  # eeg.shape=(#channel, T)
        elif all([len(x)==1 for x in all_ok_channels2]):
            eeg_channels = [x[0] for x in all_ok_channels2]
            eeg = edf.get_data(picks=eeg_channels)
            eeg = np.array([eeg[6]-eeg[1],  # F3-M2
                            eeg[7]-eeg[0],  # F4-M1
                            eeg[2]-eeg[1],  # C3-M2
                            eeg[3]-eeg[0],  # C4-M2
                            eeg[4]-eeg[1],  # O1-M2
                            eeg[5]-eeg[0],])# O2-M2

        else:
            raise ValueError(f'EEG not found. Contained channels: {edf_channels}')

    # convert to uV
    eeg = eeg*1e6
    
    return eeg, Fs, eeg_channels, start_time
    
