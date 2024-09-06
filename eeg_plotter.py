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
from reader import WORD_LEVEL_FEATURES

MAX_SIMULTANEOUS_WORDS_TO_PROCESS = 10

MISSING_CHANNELS = [126, 127, 48, 119, 17, 128, 49, 56, 63, 68, 73, 81, 88, 94, 99, 107, 113, 125, 21, 25, 32, 1, 8, 14] #ISSUE: 24 elements instead of the 23 stated in the paper


def channel_fill(data, missing_channels = MISSING_CHANNELS):
    '''
    Takes a numpy array containing the values corresponding to the 105 channels used in the paper and maps them to 
    the full 128 channels of the GSN-HydroCel-128 system (the one used to extract the data), filling the missing 
    channels with the MISSING_DATA_SYMBOL defined in reader.py. A different set of missing channels can be optionally 
    defined through the missing_channels parameter. 
    '''
    result = []
    idx = 0
    for i in range(128):
        if i+1 in missing_channels:
            result.append(MISSING_DATA_SYMBOL)
        else:
            result.append(data[idx])
            idx+=1
    return np.array(result)


def get_evokeds_for_word_data(word_df):
    evokeds = {}
    for feature in WORD_LEVEL_FEATURES:
        if np.equal(word_df[feature].to_numpy(), np.array([MISSING_DATA_SYMBOL for _ in range(EEG_CHANNEL_COUNT)])).all():
            continue # data for this feature is missing
        info = mne.create_info(['E'+str(i) for i in range(1, 129)], 500, ch_types='eeg')
        info['bads']=['E'+str(x) for x in MISSING_CHANNELS]
        data = np.expand_dims(channel_fill(word_df[feature]), -1)
        evokeds[feature] = mne.EvokedArray(data, info)
        evokeds[feature].set_montage("GSN-HydroCel-128")
    return evokeds


def process_word(word_file_path, word_plots_folder):
    word_df = pd.read_csv(word_file_path, sep='\t')
    evokeds = get_evokeds_for_word_data(word_df)
    for plotname in evokeds.keys():
        fig = evokeds[plotname].plot_topomap(times=0, show=False)
        fig.savefig(word_plots_folder+'/'+plotname+'.png', format='png', dpi=1200)
        plt.close()
    word_idx_str = word_file_path.split('/')[-1][5:-4]
    print(f'Finished processing word {word_idx_str}')


if __name__ == '__main__':
    task = 'none'
    while task not in ['NR', 'TSR']:
        task = input('Plot for which task? ("NR" or "TSR"):\n')
    
    main_path = EXTRACTED_DATA_PATH + "extracted_data_" + task + "/eeg_data"
    for subject in tqdm(os.listdir(main_path)):
        subject_path = main_path + "/" + subject
        for sent_idx_str in tqdm(os.listdir(subject_path)):
            print(f'Processing data for subject {subject}, sentence {sent_idx_str}')
            sentence_path = subject_path + "/" + sent_idx_str
            processes = []
            for word_filename in os.listdir(sentence_path):
                if word_filename[:5] == "word_" and word_filename[-4:] == ".tsv": # valid word data file
                    word_idx_str = word_filename[5:-4]
                    word_plots_folder = sentence_path + '/word_' + word_idx_str + '_plots'
                    if not os.path.exists(word_plots_folder):
                        os.mkdir(word_plots_folder)
                    procname = "sentence"+sent_idx_str+"_word"+word_idx_str
                    processes.append( mp.Process(target=process_word, args=(sentence_path+"/"+word_filename, word_plots_folder), 
                                                         name=procname) )
            index = 0
            while index < len(processes):
                batch_size = min(len(processes)-index, MAX_SIMULTANEOUS_WORDS_TO_PROCESS)
                for i in range(batch_size):
                    processes[index+i].start()
                for i in range(batch_size):
                    processes[index+i].join()
                index+=batch_size
            print(f'Finished processing data for subject {subject}, sentence {sent_idx_str}')
                    
                        
                    
