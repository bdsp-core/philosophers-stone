from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import os

cwd = os.getcwd()
path_parent = '/path/'

    
def load_master_table(csv_file):
    """
    simple function that loads the current master table. always call this function from every part of the code, guarantees compatibility.
    :return:
    dataframe (csv file)
    """
    table = pd.read_csv(csv_file, low_memory=False)
    assert all(table.exists_spec_eeg == True), f'exists_spec_eeg is not True for all subjects. {table.exists_spec_eeg.value_counts()}'

    return table



class Spectrogram_TMP_SMALL_Dataset(Dataset):
    # multitask, with sleep stages

    def __init__(self, path_mastertable=None, target_var=None, transform=None):

        table = load_master_table(path_mastertable)
        if target_var is None:
            target_var = 'cog_total'

        self.path_spec = table.path_spec_eeg.values
        self.y = list(table[target_var].values)
        self.path_stage = table.path_stage.values
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        spec = torch.load(self.path_spec[index])[:, :396, :]
        y = self.y[index]
        stage = torch.load(self.path_stage[index])[:396, :]
        if self.transform:
            spec = self.transform(spec)

        return spec, y, stage
    


class Spectrogram_1_2_Dataset(Dataset):
    # multitask, with sleep stages

    def __init__(self, path_mastertable=None, regression_targets=None, classification_targets=None,
                 transform=None, covariates=[], target_tl=None): #
                #  transfer_learning=False, tl_regression_targets=None, tl_classification_targets=None):

        table = load_master_table(path_mastertable)
        if regression_targets is None:
            regression_targets = 'cog_total'
        if classification_targets is None:
            classification_targets = 'dx-cs-dementia'

        # remove any rows where all of the regression and all of the classification targets are NaN:
        if (table[regression_targets].isnull().all(axis=1) & table[classification_targets].isnull().all(axis=1)).sum() > 0:
            print('removing rows where all regression and classification targets are NaN')
            print('shape before:', table.shape)
            table = table.dropna(subset=regression_targets + classification_targets, how='all')
            print('shape after:', table.shape)

        # set regression and classification targets.
        if target_tl is not None:
            # remove all entries where target_tl is nan:
            table_shape_preslice = table.shape
            table = table.dropna(subset=[target_tl])
            print(f'\033[31mTL target: {target_tl}\033[0m', f'Removed rows where target is nan. Shape before: {table_shape_preslice}, shape after: {table.shape}')
        table_regression_targets = table[regression_targets].copy()
        table_classification_targets = table[classification_targets].copy()
        
        if 'wavelet' in table.path_spec_eeg.iloc[0]:
            keyword_type_eeg = 'wavelet'
        elif 'timeseries' in table.path_spec_eeg.iloc[0]:
            keyword_type_eeg = 'timeseries'
        else:
            keyword_type_eeg = None

        if keyword_type_eeg is not None:
            # check if the path to the spectrograms and sleep stages exists on the internal hard drive, if so, replace the path in the master table.
            if 'sleep_oracle/data/' in table.path_spec_eeg.iloc[0]:
                path_external_drive = table.path_spec_eeg.iloc[0].split(f'{keyword_type_eeg}_')[0]
                path_folder = table.path_spec_eeg.iloc[0].split('sleep_oracle/data/')[1]
                path_folder = path_folder.split('/')[:-1]
                path_folder = '/'.join(path_folder)
                path_internal_drive = '/d/sleep_oracle_data/'
                path_internal_folder = os.path.join(path_internal_drive, path_folder)

                if 1: 
                    if os.path.exists(path_internal_folder):
                        print(f'\033[35mPath to spectrograms and sleep stages exists on internal hard drive. Replacing path in master table.\033[0m')

                        random_subset = False

                        if not random_subset:
                            # Apply the replacement to all rows
                            table.path_spec_eeg = table.path_spec_eeg.str.replace(path_external_drive, path_internal_drive)
                            table.path_stage = table.path_stage.str.replace(path_external_drive, path_internal_drive)
                        else:
                            # Determine a random 75% subset of indices and replace the path only for those. This is to not overload the drives.
                            random_subset = np.random.choice(table.index, size=int(0.75 * len(table)), replace=False)
                            table.loc[random_subset, 'path_spec_eeg'] = table.loc[random_subset, 'path_spec_eeg'].str.replace(path_external_drive, path_internal_drive)
                            table.loc[random_subset, 'path_stage'] = table.loc[random_subset, 'path_stage'].str.replace(path_external_drive, path_internal_drive)
                            # check how many entries have the external and internal drive path for path_spec_eeg:
                            n_external = table.path_spec_eeg.str.contains(path_external_drive).sum()
                            n_internal = table.path_spec_eeg.str.contains(path_internal_drive).sum()
                            print(f'Files read in through external and internal hard drive: N external: {n_external}, N internal: {n_internal}')
                            # assert that both one selected sample from internal and external exist:
                            index_internal = random_subset[0]
                            index_external = list(set(table.index) - set(random_subset))[0]
                            assert os.path.exists(table.path_spec_eeg.iloc[index_internal]), f"{table.path_spec_eeg.iloc[index_internal]} does not exist."
                            assert os.path.exists(table.path_stage.iloc[index_external]), f"{table.path_stage.iloc[index_external]} does not exist."
                            assert os.path.exists(table.path_spec_eeg.iloc[index_external]), f"{table.path_spec_eeg.iloc[index_external]} does not exist."
                            assert os.path.exists(table.path_stage.iloc[index_internal]), f"{table.path_stage.iloc[index_internal]} does not exist."

        # assert that those sample paths exist:
        assert os.path.exists(table.path_spec_eeg.iloc[len(table)//2]), f"{table.path_spec_eeg.iloc[len(table)//2]} does not exist."
        assert os.path.exists(table.path_stage.iloc[len(table)//2]), f"{table.path_stage.iloc[len(table)//2]} does not exist."

        print(f'Sample path to spectrograms: {table.path_spec_eeg.iloc[len(table)//2]}')
        print(f'Sample path to sleep stages: {table.path_stage.iloc[len(table)//2]}')

        self.path_spec = table.path_spec_eeg.values
        self.y_regression = list(table_regression_targets.values)
        self.y_classification = list(table_classification_targets.values)
        self.path_stage = table.path_stage.values
        self.transform = transform
        self.fileid = table.fileid.values

        # for the wlt and mlt models, the stage was saved in the same fs as the spectrogram, however here it is actually more convenient to
        # always have 1 Hz as resolution. Therefore, resample according to the fs of the spectrogram.
        # In the future, more efficient to save it in the 1 Hz resolution.
        fs_stage_resample = 1
        resample_map = {
            '2hz': 2,
            '4hz': 4,
            '8hz': 8
        }
        if any(x in path_mastertable for x in ['mlt', 'wlt']):
            for hz_string, fs in resample_map.items():
                if hz_string in path_mastertable:
                    fs_stage_resample = fs
                    break 
        self.fs_stage_resample = int(fs_stage_resample)
        
        # cohort one-hot encoding
        cohorts = list(pd.unique(table.cohort))
        # cohorts = ['fhs', 'mesa', 'koges', 'sof', 'mros', 'mgh-cog', 'mgh-mmse']
        df_template = pd.DataFrame(columns=cohorts)
        one_hot = pd.get_dummies(table.cohort)
        cohort_encoding = pd.concat([df_template, one_hot], axis=0)
        cohort_encoding.fillna(0, inplace=True)
        cohort_encoding = cohort_encoding.values
        cohort_encoding = cohort_encoding.astype(np.float32)
        assert all(cohort_encoding.sum(axis=1) == 1)
        assert cohort_encoding.shape[1] == len(cohorts)
        self.cohort_encoding = torch.Tensor(cohort_encoding)

        self.sex = torch.Tensor(table.sex.values[:, np.newaxis])
        assert all(np.isin(self.sex, [0, 1, 0.5]))  # 0: female, 1: male, 0.5: unknown/non-binary

        self.age_z = torch.Tensor(table.age_z.values[:, np.newaxis])

        covariates_tensor = torch.Tensor()
        if 'cohort' in covariates:
            # concat covariates_tensor with cohort encoding
            covariates_tensor = torch.cat([covariates_tensor, self.cohort_encoding], dim=1)
        if 'age' in covariates:
            covariates_tensor = torch.cat([covariates_tensor, self.age_z], dim=1)
        if 'sex' in covariates:
            covariates_tensor = torch.cat([covariates_tensor, self.sex], dim=1)
        if 'apoe' in covariates:
            print('apoe not implemented yet')
        self.covariates = covariates_tensor

        assert len(self.path_spec) == len(self.y_regression) == len(self.y_classification) == len(self.path_stage) == len(self.cohort_encoding) == len(self.sex), \
        f"{len(self.path_spec)}, {len(self.y_regression)}, {len(self.y_classification)}, {len(self.path_stage)}, {len(self.cohort_encoding)}, {len(self.sex)}"

        self.load_spec = self._default_load_spec
            
    def __len__(self):
        return len(self.y_regression)

    def _load_spec_with_transformation(self, index):
        """ Default method to load the spectrogram and apply a transformation (shift by +- 3 seconds). 
        specs shape: torch.Size([1, 39600 * fs, 100])  (B, T, F)
        """
        fs = self.fs_stage_resample
        spec_data = torch.load(self.path_spec[index])
        # shift T dimension by +- 3 seconds
        shift = np.random.randint(-3*fs, 3*fs)
        # if shift is negative, remove the first shift elements, repeat last elements at the end. 
        # if shift is positive, remove the last shift elements, repeat first elements at the beginning.
        if shift < 0:
            spec_data = torch.cat((spec_data[:, -shift:, :], spec_data[:, shift:, :]), dim=1)
        elif shift > 0:
            spec_data = torch.cat((spec_data[:, :shift, :], spec_data[:, :-shift, :]), dim=1)
        else:
            pass

        # random amplitude scaling by factor 0.99 to 1.01:
        spec_data = spec_data * (0.99 + 0.02 * torch.rand(1))

        # random offset between -0.1 and 0.1:
        spec_data = spec_data + 0.2 * torch.rand(1) - 0.1

        return spec_data

    def _default_load_spec(self, index):
        """ Default method to load the spectrogram. """
        # acutally call _load_spec_ts_with_transformation:
        return self._load_spec_with_transformation(index)
    
    def _mwlt_load_spec2d(self, index):
        """ Load both mlt and wlt spectrograms. """
        spec_mlt = torch.load(self._load_spec_with_transformation[self.path_mlt_spec[index]])
        spec_wlt = torch.load(self._load_spec_with_transformation(self.path_wlt_spec[index]))
        combined_spec = torch.cat((spec_mlt, spec_wlt), dim=0)
        return combined_spec
    
    def _mwlt_load_spec3d(self, index):
        """ Load both mlt and wlt spectrograms. """
        spec_mlt = self._load_spec_with_transformation(self.path_mlt_spec[index])
        spec_wlt = self._load_spec_with_transformation(self.path_wlt_spec[index])
        spec_wlt_ss = self._load_spec_with_transformation(self.path_wlt_ss_spec[index])
        combined_spec = torch.cat((spec_mlt, spec_wlt, spec_wlt_ss), dim=0)
        return combined_spec
    
    def _load_spec_ts_with_transformation(self, index):
        """ Default method to load the timeseries data and apply a transformation (shift by +- 3 seconds). 
        (size spec: torch.Size([1, 7920000])) """
        fs = 200
        ts_data_original = torch.load(self.path_spec[index])
        shift = np.random.randint(-3*fs, 3*fs)
        # if shift is negative, remove the first shift elements, repeat last elements at the end.
        # if shift is positive, remove the last shift elements, repeat first elements at the beginning.
        if shift < 0:
            ts_data = torch.cat((ts_data_original[:, -shift:], ts_data_original[:, shift:]), dim=1)
        elif shift > 0:
            ts_data = torch.cat((ts_data_original[:, :shift], ts_data_original[:, :-shift]), dim=1)
        else:
            ts_data = ts_data_original
        
        # random amplitude scaling by factor 0.99 to 1.01:
        ts_data = ts_data * (0.99 + 0.02 * torch.rand(1))
        
        # random offset between -0.1 and 0.1:
        ts_data = ts_data + 0.2 * torch.rand(1) - 0.1
        
        return ts_data
    
    def __getitem__(self, index):

        try:
            spec = self.load_spec(index)
            y_regression = self.y_regression[index]
            y_classification = self.y_classification[index]
            stage = torch.load(self.path_stage[index])[::self.fs_stage_resample, :]
            if self.transform:
                spec = self.transform(spec)

            covariates = self.covariates[index]
        except Exception as e:
            print(f"Error with index {index}: {e}")
            print(f'dataloader, index: {index}, self.path_spec: {self.path_spec[index]}, self.path_stage: {self.path_stage[index]}')
            raise e

        return spec, y_regression, y_classification, stage, covariates
    