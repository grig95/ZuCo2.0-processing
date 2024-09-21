import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import gc
from wordfreq import word_frequency

from reader import EXTRACTED_DATA_PATH
from reader import EEG_CHANNEL_COUNT
from reader import MISSING_DATA_SYMBOL
from reader import EEG_FEATURES
from reader import ET_FEATURES

from reader import get_subjects_list

MISSING_CHANNELS = [126, 127, 48, 119, 17, 128, 49, 56, 63, 68, 73, 81, 88, 94, 99, 107, 113, 125, 21, 25, 32, 1, 8, 14] #ISSUE: 24 elements instead of the 23 stated in the paper

LANGUAGE_CHANNELS = [33, 39, 45, 50, 27, 34, 40, 46, 51] # language-relevant channels in full electrode map indices

TRT_RANGES = [(150, 15), (250, 25), (500, 50), (1000, 100)] # (upper_limit, time_step)


def device_channel_to_data_channel(index):
    '''
    Takes an index between 1 and 128 corresponding to one of the channels of the GSN-HydroCel-128 system and returns  its corresponding index in the data (0-104).
    '''
    if index in MISSING_CHANNELS:
        return None
    c = 0
    for miss in MISSING_CHANNELS:
        if miss < index:
            c+=1
    return index-c-1 


def channel_fill(data, missing_channels = MISSING_CHANNELS):
    '''
    Takes a numpy array containing the values corresponding to the 105 channels used in the paper and maps them to 
    the full 128 channels of the GSN-HydroCel-128 system (the one used to extract the data), filling the missing 
    channels with the MISSING_DATA_SYMBOL defined in reader.py. A different set of missing channels can be optionally 
    defined through the missing_channels parameter. Also works with (n, 105) shaped arrays.
    '''
    mask = np.ones(128, dtype=bool)
    mask[np.array(missing_channels)-1] = False
    if len(data.shape)==1:
        if len(missing_channels)>23: # TODO: remove these two lines once the missing channels issue is resolved
            data=data[:-1]
        result = np.full(128, MISSING_DATA_SYMBOL, dtype=np.float64)
        result[mask] = data
    else: # (n, 105) shaped array
        if len(missing_channels)>23: # TODO: remove these two lines once the missing channels issue is resolved
            data=data[:, :-1]
        result = np.full((data.shape[0], 128), MISSING_DATA_SYMBOL, dtype=np.float64)
        result[:, mask] = data
    return result


def get_evoked_for_eeg_data(eeg_data, frequency=500): # TODO freq: The frequency argument is only here to circumvent the weird sample count relative to TRT in the generate_trt_statistic function. Remove/do something with it once you figure out what's going on.
    '''
    Takes in a (105,) or (n, 105) shaped numpy array and returns the associated evoked object for plotting.
    '''
    info = mne.create_info(['E'+str(i) for i in range(1, 129)], frequency, ch_types='eeg')
    info['bads']=['E'+str(x) for x in MISSING_CHANNELS]
    if len(eeg_data.shape)==1:
        data = np.expand_dims(channel_fill(eeg_data), -1)
    else:
        data = np.transpose(channel_fill(eeg_data))
    evoked = mne.EvokedArray(data*1e-6, info) # EvokedArray expects the data to be measured in volts, and the dataset measures it in microvolts
    evoked.set_montage("GSN-HydroCel-128")
    return evoked


def get_raw_word_eeg(args):
    '''
    args = (task, subject, sentence_id, word_idx)
    '''
    task, subject, sentence_id, word_idx = args
    word_path = EXTRACTED_DATA_PATH + 'extracted_data_' + task + '/eeg_data/' + subject + '/' + str(sentence_id) + '/word_' + str(word_idx) + '_raw.tsv'
    if not os.path.exists(word_path): # Some words are fixated but have missing eeg data. These are marked in the dataset by a [[nan]] array and reader.py does not save tsv files for them.
        return None
    word_df = pd.read_csv(word_path, sep='\t', usecols=lambda col: col!='sentence_relative_fix_idx')
    eeg_array = word_df.to_numpy(dtype=np.float64)
    return eeg_array

def get_raw_word_eeg_mean(args):
    '''
    args = (task, subject, sentence_id, word_idx)
    '''
    data = get_raw_word_eeg(args)
    if data is None:
        return None
    return np.mean(data, axis=0)


def generate_median_relative_mean_plots_for_subject(task, subject, feature):
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


def helper_get_len_normalized_raw_eeg(args):
    task, subject, sentence_id, word_idx, target_length = args
    raw_eeg = get_raw_word_eeg((task, subject, sentence_id, word_idx))
    if raw_eeg is None:
        return None
    result = np.empty((target_length, raw_eeg.shape[1]), dtype=np.float64)
    min_len = min(target_length, raw_eeg.shape[0])
    result[:min_len, :] = raw_eeg[:min_len, :]
    if min_len < target_length:
        result[min_len:, :] = np.mean(raw_eeg, axis=0)
    return result


def generate_mean_eeg_plots_for_trt_range(lower_bound, upper_bound, time_step, task):
    high_lvl_data = pd.read_csv(EXTRACTED_DATA_PATH+'/extracted_data_'+task+'/data.tsv', sep='\t')
    relevant_df = high_lvl_data[ (high_lvl_data['TRT']>lower_bound) & (high_lvl_data['TRT']<=upper_bound) ] [ ['subject', 'sentence_id', 'word_idx'] ]
    sum = np.zeros((upper_bound, EEG_CHANNEL_COUNT), dtype=np.float64)
    count=0
    for args in zip([task for _ in range(relevant_df.shape[0])], relevant_df['subject'], 
                    relevant_df['sentence_id'], relevant_df['word_idx'], [upper_bound for _ in range(relevant_df.shape[0])]):
        eeg_data = helper_get_len_normalized_raw_eeg(args)
        if eeg_data is None:
            continue
        sum += eeg_data
        count+=1
    mean_eeg = sum/count
    plots_folder = f'{EXTRACTED_DATA_PATH}extracted_data_{task}'
    evoked = get_evoked_for_eeg_data(mean_eeg, frequency=1000) #TODO freq: The frequency is set here only because of the sample count weirdness. Figure it out!
    fig = evoked.plot_topomap(times=[t/1000 for t in range(0, upper_bound, time_step)], show=False)
    fig.savefig(f'{plots_folder}/TRT_{lower_bound}-{upper_bound}_mean_topomap.png', format='png', dpi=1200)
    plt.close()
    # constructing the plot for the language relevant electrodes
    picks = [f'E{c}' for c in LANGUAGE_CHANNELS]
    times = evoked.times
    data = evoked.copy().pick(picks).data
    plt.figure(figsize=(10, 6))
    for i, channel in enumerate(picks):
        plt.plot(times, data[i, :], label=channel)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'EEG Data from Channels: {", ".join(picks)}')
    plt.axvline(0, color='black', linestyle='--')  # Add vertical line at 0s (stimulus onset)
    plt.legend(loc='upper right')
    plt.savefig(f'{plots_folder}/TRT_{lower_bound}-{upper_bound}_mean_langchan.png', format='png', dpi=1200)
    plt.close()


def helper_get_eeg_mean_data_in_trt_range_for_word(args):
    word, task, subject, sentence_id, word_idx, lower_bound, upper_bound = args
    eeg_path = f'{EXTRACTED_DATA_PATH}extracted_data_{task}/eeg_data/{subject}/{sentence_id}/word_{word_idx}_raw.tsv'
    if not os.path.exists(eeg_path):
        return None
    eeg_df = pd.read_csv(eeg_path, sep='\t', usecols=lambda col: col!='sentence_relative_fix_idx')
    eeg_data = eeg_df.to_numpy(dtype=np.float64)
    lower_bound = max(0, lower_bound) 
    upper_bound = min(upper_bound, len(eeg_data)) #TODO freq: the len(eeg_data) on this line is only correct if a sample rate of 1000Hz is assumed, see eeg sampling weirdness issue.
    eeg_mean = np.mean(eeg_data[lower_bound:upper_bound, :], axis=0)
    return (word, eeg_mean)



def get_input_from_list(accepted_input, msg=None):
    if msg is None:
        msg = f'Choose one from {accepted_input}:\n'
    val = None
    while val not in accepted_input:
        val = input(msg)
    return val


if __name__ == '__main__':
    msg = '''What to do?
1. Plot the ET feature related, per-subject, median-relative EEG means for a given task.
2. Compute and show a histogram of the distribution of TRT values for a given task. 
3. Plot mean EEG data for TRT ranges.
4. Plot the mean activation between 200-400ms of each language-relevant electrode for words fixated longer than 400ms against the word's frequency.
5. Plot the word frequency distribution of the whole dataset.
'''
    option = get_input_from_list(['1', '2', '3', '4', '5', 'DEBUG'], msg)
    
    if option=='1':
        task = get_input_from_list(['NR', 'TSR'], 'Plot for which task? "NR" or "TSR":\n')
        et_feature = get_input_from_list(ET_FEATURES+['all'], f'Plot for which feature? One of {ET_FEATURES+["all"]}:\n')
        subject_list = get_subjects_list()
        subject = get_input_from_list(subject_list+['all'], f'Plot for which subject? One of {subject_list+["all"]}:\n')
        
        if subject == 'all':
            if et_feature == 'all':
                features = ET_FEATURES
            else:
                features = [et_feature]
            for feature in features:
                print(f'Plotting feature {feature} for all subjects. Existing plots will be overriden. Plots will not be automatically displayed in this mode.')
                for sub in subject_list:
                    print(f'Plotting for subject {sub}...')
                    generate_median_relative_mean_plots_for_subject(task, sub, feature)
                    print(f'Finished plotting for subject {sub}...')
            print('Plots saved. To view them run this script and pick a single subject and a single feature.')
        else:
            print(f'Plotting feature {et_feature} for subject {subject}. Plot displaying not yet supported, please check the files.')
            generate_median_relative_mean_plots_for_subject(task, subject, et_feature)
            print('Finished')
    
    elif option=='2':
        task = get_input_from_list(['NR', 'TSR'], 'Plot for which task? "NR" or "TSR":\n')
        data_path=EXTRACTED_DATA_PATH+'extracted_data_'+task+'/data.tsv'
        data_df = pd.read_csv(data_path, sep='\t')
        bins = [x for x in range(int(min(data_df['TRT'].dropna())), int(max(data_df['TRT'].dropna())+1))]
        plt.hist(data_df['TRT'], bins=bins)
        plt.xlabel('TRT values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of TRT values for task {task}')
        plt.show()
    
    elif option=='3':
        task = get_input_from_list(['NR', 'TSR'], 'Plot for which task? "NR" or "TSR":\n')
        lower_bound = 0
        for trt_range in TRT_RANGES:
            print(f'Plotting for TRT values {lower_bound}-{trt_range[0]}ms...')
            generate_mean_eeg_plots_for_trt_range(lower_bound, trt_range[0], trt_range[1], task)
            lower_bound=trt_range[0]
        print('Done')
    
    elif option=='4':
        lower_trt_bound = 200
        upper_trt_bound = 400
        task = get_input_from_list(['NR', 'TSR'], 'Plot for which task? ("NR" or "TSR"):\n')
        task_folder = f'{EXTRACTED_DATA_PATH}extracted_data_{task}'
        high_df = pd.read_csv(f'{task_folder}/data.tsv', sep='\t')
        relevant_df = high_df[ (high_df['TRT']>=400) & (~high_df['content'].isna()) ] [['subject', 'sentence_id', 'word_idx', 'content', 'TRT']]
        print('Gathering and processing data...')
        with mp.Pool() as pool:
            words_and_means = pool.map(helper_get_eeg_mean_data_in_trt_range_for_word, 
                                       zip( relevant_df['content'], [task for _ in range(relevant_df.shape[0])], relevant_df['subject'], 
                                            relevant_df['sentence_id'], relevant_df['word_idx'], 
                                            [lower_trt_bound for _ in range(relevant_df.shape[0])], [upper_trt_bound for _ in range(relevant_df.shape[0])]) )
        print('Arranging data...')
        wfreqs = []
        means = []
        for wm in words_and_means:
            if wm is None:
                continue
            wfreqs.append(word_frequency(wm[0], 'en'))
            means.append(np.expand_dims(wm[1], axis=0))
        means=np.concatenate(means, axis=0)
        for channel in LANGUAGE_CHANNELS:
            print(f'Generating plot for E{channel}...')
            channel_idx = device_channel_to_data_channel(channel)
            plt.figure(figsize=(10, 6))
            plt.plot(means[:, channel_idx], wfreqs, '.b') # plot with blue dots
            plt.xlabel(f'Activation of E{channel} (µV)')
            plt.ylabel('Word frequency')
            plt.title(f'Word frequency vs. E{channel} mean activation ({lower_trt_bound}-{upper_trt_bound}ms) graph')
            plt.savefig(f'{EXTRACTED_DATA_PATH}extracted_data_{task}/wordfreq_vs_E{channel}_mean_for_TRT{lower_trt_bound}-{upper_trt_bound}.png', format='png', dpi=1200)
            plt.close()
        print('Done')
    
    elif option=='5':
        task = get_input_from_list(['NR', 'TSR'], 'Plot for which task? ("NR" or "TSR"):\n')
        df = pd.read_csv(f'{EXTRACTED_DATA_PATH}extracted_data_{task}/data.tsv', sep='\t')
        df = df[ ~df['content'].isna() ]
        freqdict = {}
        for word in df['content']:
            freq = word_frequency(word, 'en')
            if freq not in freqdict:
                freqdict[freq]=1
            else:
                freqdict[freq]+=1
        xfreq = []
        ycount = []
        for freq in freqdict:
            xfreq.append(freq)
            ycount.append(freqdict[freq])
        plt.figure(figsize=(10, 6))
        plt.plot(xfreq, ycount, '.b') # plot with blue dots
        plt.xlabel(f'Word frequency')
        plt.ylabel('Number of occurences')
        plt.title(f'Word frequency vs. word frequency occurence count graph')
        plt.show()
        plt.close()
    
    elif option == 'DEBUG':
        pass