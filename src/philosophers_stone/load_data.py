import numpy as np 
import pandas as pd
import os
import h5py



def load_prepared_data(file_path:str, signals_to_load:list=None, annotations_to_load:list=None):
    """ 
    load prepared data format (post March 22, 2023). Currently, all signals are read. 
    Inputs:
    file_path: path to prepared .h5 file.
    Returns signals: pd.DataFrame of signals.
    Returns annotations: pd.DataFrame of annotations.
    Usual coding:
    -- Sleep stages: 0/9=unstaged/unknown, 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake.
    -- Arousal: 0=none, 1=arousal.
    -- Leg movement: 0=none, 1=leg movement.
    -- Respiratory event: 0=none, 1=obstructive apnea, 2=central apnea, 3=mixed apnea, 4=hypopnea, 5=RERA.
    Returns params: dict of parameters (fs, unit_voltage).
    """

    signals = pd.DataFrame([])
    annotations = pd.DataFrame([])

    with h5py.File(file_path, "r") as f:
        
        group = f['signals']
        signals_contained = list(group.keys())
        # if 'c4-m1' is not in signals_to_load but only "eeg" is in signals_contained, then we load "eeg

        if signals_to_load is None:
            signals_to_load = signals_contained
            rename_eeg_to_c4m1 = False
            if 'c4-m1' not in signals_contained and 'eeg' in signals_contained:
                rename_eeg_to_c4m1 = True
                signals_to_load = [x.replace('c4-m1', 'eeg') for x in signals_to_load]
            dataset_names = signals_contained
        else:
            rename_eeg_to_c4m1 = False
            if 'c4-m1' in signals_to_load and ('eeg' in signals_contained and 'c4-m1' not in signals_contained):
                # replace 'c4-m1' with 'eeg':
                rename_eeg_to_c4m1 = True
                signals_to_load = [x.replace('c4-m1', 'eeg') for x in signals_to_load]
            dataset_names = signals_to_load
                
        for dataset_name in dataset_names:
            try:
                dataset = group[dataset_name][:]
            except Exception as e:
                raise ValueError(f"Dataset {dataset_name} not found in file {file_path}. Available datasets: {signals_contained}") from e
                
            assert dataset.shape[1] == 1, "Only one-dimensional datasets expected"
            signals[dataset_name] = dataset.flatten()
        if rename_eeg_to_c4m1:
            signals.rename(columns={'eeg': 'c4-m1'}, inplace=True)
        
        if 'annotations' in f.keys():
            group = f['annotations']
            if annotations_to_load is None:
                dataset_names = list(group.keys())
            else:
                dataset_names = annotations_to_load
            for dataset_name in dataset_names:
                dataset = group[dataset_name][:]
                assert dataset.shape[1] == 1, "Only one-dimensional datasets expected"
                annotations[dataset_name] = dataset.flatten()

        params = {}
        params['fs'] = f.attrs['sampling_rate']
        params['unit_voltage'] = f.attrs['unit_voltage']

        signals = signals.astype(float)
        annotations = annotations.astype(float)
        
        # print(pd.DataFrame(np.abs(signals)).describe())

    return signals, annotations, params

