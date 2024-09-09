import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from reader import EXTRACTED_DATA_PATH
from reader import EEG_CHANNEL_COUNT
from reader import MISSING_DATA_SYMBOL
from reader import EEG_FEATURES
from reader import ET_FEATURES

from reader import get_subjects_list

MISSING_CHANNELS = [126, 127, 48, 119, 17, 128, 49, 56, 63, 68, 73, 81, 88, 94, 99, 107, 113, 125, 21, 25, 32, 1, 8, 14] #ISSUE: 24 elements instead of the 23 stated in the paper


def channel_fill(data, missing_channels = MISSING_CHANNELS):
    '''
    Takes a numpy array containing the values corresponding to the 105 channels used in the paper and maps them to 
    the full 128 channels of the GSN-HydroCel-128 system (the one used to extract the data), filling the missing 
    channels with the MISSING_DATA_SYMBOL defined in reader.py. A different set of missing channels can be optionally 
    defined through the missing_channels parameter. 
    '''
    if len(missing_channels)>23: # TODO: remove these two lines once the missing channels issue is resolved
        data=data[:-1]
    mask = np.ones(128, dtype=bool)
    mask[np.array(missing_channels)-1] = False
    result = np.full(128, MISSING_DATA_SYMBOL, dtype=np.float64)
    result[mask] = data
    return result


def get_evoked_for_eeg_data(eeg_data):
    '''
    Takes in a (105,) shaped numpy array and returns the associated evoked object for plotting.
    '''
    info = mne.create_info(['E'+str(i) for i in range(1, 129)], 500, ch_types='eeg')
    info['bads']=['E'+str(x) for x in MISSING_CHANNELS]
    data = np.expand_dims(channel_fill(eeg_data), -1)
    evoked = mne.EvokedArray(data, info)
    evoked.set_montage("GSN-HydroCel-128")
    return evoked


def get_raw_word_eeg_mean(args):
    task, subject, sentence_id, word_idx = args
    word_path = EXTRACTED_DATA_PATH + 'extracted_data_' + task + '/eeg_data/' + subject + '/' + str(sentence_id) + '/word_' + str(word_idx) + '_raw.tsv'
    word_df = pd.read_csv(word_path, sep='\t')
    eeg_array = word_df.to_numpy(dtype=np.float64)
    if eeg_array.shape[1] != EEG_CHANNEL_COUNT: # Some words are fixated but have missing eeg data. These are marked by a [[nan]] array.
        return None
    return np.mean(eeg_array, axis=0)


def generate_plots_for_subject(task, subject, feature):
    high_lvl_df = pd.read_csv(EXTRACTED_DATA_PATH+'extracted_data_'+task+'/data.tsv', sep='\t')
    subject_df = high_lvl_df[(high_lvl_df['subject']==subject) & (~high_lvl_df[feature].isna())] [['sentence_id', 'word_idx', feature]]
    subject_path = EXTRACTED_DATA_PATH + 'extracted_data_' + task + '/eeg_data/' + subject
    median = np.median( subject_df[feature].to_numpy(dtype=np.float64) )
    with mp.Pool() as pool:
        below_median_df = subject_df[ subject_df[feature] < median ]
        below_median = pool.map(get_raw_word_eeg_mean, zip([task for _ in range(below_median_df.shape[0])], [subject for _ in range(below_median_df.shape[0])], 
                                          below_median_df['sentence_id'], below_median_df['word_idx']))
        # remove invalid data and calculate mean
        below_median = [data for data in below_median if data is not None]
        below_median_mean = np.mean(np.array(below_median, dtype=np.float64), axis=0)
        below_median_mean_evoked = get_evoked_for_eeg_data(below_median_mean)
        # generate plot
        fig = below_median_mean_evoked.plot_topomap(times=0, show=False)
        fig.savefig(subject_path+'/'+feature+'_below_median_mean.png', format='png', dpi=1200)
        plt.close()
    with mp.Pool() as pool:
        above_median_df = subject_df[ subject_df[feature] > median ]
        above_median = pool.map(get_raw_word_eeg_mean, zip([task for _ in range(above_median_df.shape[0])], [subject for _ in range(above_median_df.shape[0])], 
                                          above_median_df['sentence_id'], above_median_df['word_idx']))
        # remove invalid data and calculate mean
        above_median = [data for data in above_median if data is not None]
        above_median_mean = np.mean(np.array(above_median, dtype=np.float64), axis=0)
        above_median_mean_evoked = get_evoked_for_eeg_data(above_median_mean)
        fig = above_median_mean_evoked.plot_topomap(times=0, show=False)
        fig.savefig(subject_path+'/'+feature+'_above_median_mean.png', format='png', dpi=1200)
        plt.close()


if __name__ == '__main__':
    task = 'none'
    while task not in ['NR', 'TSR']:
        task = input('Plot for which task? "NR" or "TSR":\n')

    et_feature = 'none'
    while et_feature not in ET_FEATURES+['all'] and et_feature!='SFD': # TODO: remove the sfd part once data_loading_helpers.extract_word_level_data is modified to fetch it
        et_feature = input(f'Plot for which feature? One of {ET_FEATURES+["all"]}:\n')
    
    subject = 'none'
    subject_list = get_subjects_list()
    while subject not in subject_list+['all']:
        subject = input(f'Plot for which subject? One of {subject_list+["all"]}:\n')
    
    if subject == 'all':
        if et_feature != 'all':
            print(f'Plotting feature {et_feature} for all subjects. Existing plots will be overriden. Plots will not be automatically displayed in this mode.')
            for sub in subject_list:
                print(f'Plotting for subject {sub}...')
                generate_plots_for_subject(task, sub, et_feature)
                print(f'Finished plotting for subject {sub}...')
        else:
            for feature in ET_FEATURES:
                if feature != 'SFD': # TODO: remove the sfd part once data_loading_helpers.extract_word_level_data is modified to fetch it
                    print(f'Plotting feature {feature} for all subjects. Existing plots will be overriden. Plots will not be automatically displayed in this mode.')
                    for sub in subject_list:
                        print(f'Plotting for subject {sub}...')
                        generate_plots_for_subject(task, sub, feature)
                        print(f'Finished plotting for subject {sub}...')
        print('Plots saved. To view them run this script and pick a single subject.')
    else:
        print(f'Plotting feature {et_feature} for subject {subject}. Plot displaying not yet supported, please check the files.')
        generate_plots_for_subject(task, subject, et_feature)
        print('Finished')
