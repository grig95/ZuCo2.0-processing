import os
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import data_loading_helpers as dh
import multiprocessing as mp

MAX_SIMULTANEOUS_SUBJECTS_TO_PROCESS = 5

EXTRACTED_DATA_PATH = './'
NR_FILES_PATH = 'task1 - NR/Matlab files/'
TSR_FILES_PATH = 'task2 - TSR/Matlab files/'

EEG_CHANNEL_COUNT = 105
MISSING_DATA_SYMBOL = -1

SENTENCE_LEVEL_MEANS = ['mean_a1', 'mean_a2', 'mean_b1', 'mean_b2', 'mean_g1', 'mean_g2', 'mean_t1', 'mean_t2']
EEG_FEATURES = ['FFD_a1', 'FFD_a2', 'FFD_b1', 'FFD_b2', 'FFD_g1', 'FFD_g2', 'FFD_t1', 'FFD_t2',
                        'GD_a1', 'GD_a2', 'GD_b1', 'GD_b2', 'GD_g1', 'GD_g2', 'GD_t1', 'GD_t2',
                        'GPT_a1', 'GPT_a2', 'GPT_b1', 'GPT_b2', 'GPT_g1', 'GPT_g2', 'GPT_t1', 'GPT_t2',
                        'SFD_a1', 'SFD_a2', 'SFD_b1', 'SFD_b2', 'SFD_g1', 'SFD_g2', 'SFD_t1', 'SFD_t2',
                        'TRT_a1', 'TRT_a2', 'TRT_b1', 'TRT_b2', 'TRT_g1', 'TRT_g2', 'TRT_t1', 'TRT_t2'
                        ]
ET_FEATURES = ['GD', 'TRT', 'FFD', 'SFD', 'GPT']


def get_subject(filename):
    fis = os.path.basename(filename)
    return fis.split("sults")[1].split("_")[0]

def get_task(filename):
    fis = os.path.basename(filename)
    return fis.split("_")[1].replace(".mat", '')

def get_files(task = 'NR'):
    if task == 'NR' or task == 1:
        task = 'NR'
        rootdir = NR_FILES_PATH
    elif task == 'TSR' or task == 2:
        task = 'TSR'
        rootdir = TSR_FILES_PATH
    else:
        return None
    all_files = []
    for file in os.listdir(rootdir):
        if file.endswith(task+".mat"):
            file_path = os.path.join(rootdir, file)
            subject = get_subject(file_path)
            # exclude YMH due to incomplete data because of dyslexia
            if subject == 'YMH':
                continue
            all_files.append(file_path)
    return all_files


def get_subjects_list():
    return [get_subject(file) for file in get_files()]


def get_basic_data_from_file(file, columns=['content', 'word_idx'] + ET_FEATURES + ['nFix', 'reading_order']):
    subject = get_subject(file)
    task = get_task(file)
    print('subject ', subject, ' file ', file, ' task', task)

    f = h5py.File(file)
    sentence_data = f['sentenceData']
    rawData = sentence_data['rawData']
    contentData = sentence_data['content']
    omissionR = sentence_data['omissionRate']
    wordData = sentence_data['word']
    
    items = []
    for sent_idx in tqdm(range(len(rawData))):
        obj_reference_content = contentData[sent_idx][0]
        sent = dh.load_matlab_string(f[obj_reference_content])
    
        # get word level data
        word_objects = f[wordData[sent_idx][0]]
        available_objects = list(word_objects)
        if 'content' not in available_objects:
            #print(sent_idx)
            continue
        word_data = dh.extract_word_level_data(f, word_objects)
        
        for widx in range(0, len(word_data)):
            row = {'sentence': sent,
                   'sentence_id': sent_idx,
                   'subject': subject,
                   'task': task
                  }
            for col in columns:
                row[col] = word_data[widx].get(col, np.nan)
            
            # add a column for the first fixation time (by adding up the durations of all fixations preceding it)
            fixationIndices = f[ f[ wordData[sent_idx][0] ]['fixPositions'] [widx][0] ] [:]
            if len(fixationIndices.shape) != 2:
                row['first_fixation_time'] = None
            else:
                firstFixationIndex = int(np.min(fixationIndices))-1 # fixation indices start at 1 
                fixationDurations = f[ sentence_data['allFixations'][sent_idx][0] ]['duration'][:, 0]
                if firstFixationIndex >= len(fixationDurations):
                    row['first_fixation_time'] = None
                elif firstFixationIndex == 0:
                    row['first_fixation_time'] = 0.
                else:
                    row['first_fixation_time'] = np.cumsum(fixationDurations)[firstFixationIndex-1]
            
            items.append(row)
    return items


def get_eeg_data_from_file(file):
    subject=get_subject(file)
    task=get_task(file)
    f=h5py.File(file)
    sentence_data=f['sentenceData']
    wordData = sentence_data['word']
    means = SENTENCE_LEVEL_MEANS
    word_level_features = EEG_FEATURES
    data = {}
    for sent_idx in tqdm(range(len(wordData))):
        try:
            if 'content' not in list( f[wordData[sent_idx][0]].keys() ):
                continue
        except: # apparently some sentence data just doesn't exist and is represented by 1x1 NaN datasets instead of groups, this is here to catch that
            continue
        data[sent_idx]={}
        # getting means
        data[sent_idx]['means'] = {}
        # -> tagging means data
        data[sent_idx]['means']['task'] = [task for _ in range(EEG_CHANNEL_COUNT)]
        data[sent_idx]['means']['subject'] = [subject for _ in range(EEG_CHANNEL_COUNT)]
        data[sent_idx]['means']['sentence_id'] = [sent_idx for _ in range(EEG_CHANNEL_COUNT)]
        for mean in means:
            data[sent_idx]['means'][mean] = np.squeeze(f[sentence_data[mean][sent_idx][0]])
        # getting word data
        data[sent_idx]['word_data'] = {}
        for word_idx in tqdm(range(len( f[wordData[sent_idx][0] ]['content'] ))):
            data[sent_idx]['word_data'][word_idx] = {}
            # -> tagging word data
            data[sent_idx]['word_data'][word_idx]['task'] = [task for _ in range(EEG_CHANNEL_COUNT)]
            data[sent_idx]['word_data'][word_idx]['subject'] = [subject for _ in range(EEG_CHANNEL_COUNT)]
            data[sent_idx]['word_data'][word_idx]['sentence_id'] = [sent_idx for _ in range(EEG_CHANNEL_COUNT)]
            data[sent_idx]['word_data'][word_idx]['word_id'] = [word_idx for _ in range(EEG_CHANNEL_COUNT)]
            # getting et related eeg features
            for feature in word_level_features:
                # it would seem some sentences have missing features (ex: subject YHS, sentence 154 has no SFD-related feature)
                if feature in list(f[ wordData[sent_idx][0] ].keys()):
                    feature_data = np.squeeze(f[ f[ wordData[sent_idx][0] ][feature] [word_idx][0] ])
                    if len(feature_data) == EEG_CHANNEL_COUNT:
                        data[sent_idx]['word_data'][word_idx][feature] = feature_data
                    else: # some features don't have the right number of dimensions
                        data[sent_idx]['word_data'][word_idx][feature] = [MISSING_DATA_SYMBOL for _ in range(EEG_CHANNEL_COUNT)] # marking empty features
                else:
                    data[sent_idx]['word_data'][word_idx][feature] = [MISSING_DATA_SYMBOL for _ in range(EEG_CHANNEL_COUNT)] # marking empty features
            # getting raw eeg data
            if 'rawEEG' in list(f[wordData[sent_idx][0]].keys()): # again, some sentences miss the rawEEG data
                if len( f[f[wordData[sent_idx][0]]['rawEEG'][word_idx][0]].shape ) == 2: # some words have invalid data (probably the ones that were never fixated)
                    data[sent_idx]['word_data'][word_idx]['raw_eeg'] = f[ f[f[wordData[sent_idx][0]]['rawEEG'][word_idx][0]] [0][0] ] [:]
                else:
                    data[sent_idx]['word_data'][word_idx]['raw_eeg'] = np.array([[MISSING_DATA_SYMBOL for j in range(EEG_CHANNEL_COUNT)]] , dtype=np.float64) # marking missing data
            else:
                data[sent_idx]['word_data'][word_idx]['raw_eeg'] = np.array([[MISSING_DATA_SYMBOL for j in range(EEG_CHANNEL_COUNT)]] , dtype=np.float64) # marking missing data
    return data


def process_file(file, eeg_path):
    subject = get_subject(file)
    eeg_data = get_eeg_data_from_file(file)
    subject_path = eeg_path+'/'+subject
    if not os.path.exists(subject_path):
        os.mkdir(subject_path)
    for sent_idx in eeg_data.keys():
        sentence_path = subject_path+'/'+str(sent_idx)
        if not os.path.exists(sentence_path):
            os.mkdir(sentence_path)
        try:
            means_df = pd.DataFrame(eeg_data[sent_idx]['means'])
            means_df.to_csv(sentence_path+'/means.tsv', sep='\t')
        except Exception as e:
            msg = f'Extraction error at means for subject {subject}, sentence {sent_idx}:\n{e}\n'
            print(msg)
            #exception_log.write(msg)
        for word_idx in eeg_data[sent_idx]['word_data'].keys():
            try:
                word_df = pd.DataFrame({key: value for (key, value) in eeg_data[sent_idx]['word_data'][word_idx].items() if key not in ['raw_eeg']})
                word_df.to_csv(sentence_path+'/word_'+str(word_idx)+'.tsv', sep='\t')
            except Exception as e:
                msg = f'Extraction error for subject {subject}, sentence {sent_idx}, word {word_idx}:\n{e}\n'
                print(msg)
                #exception_log.write(msg)
            # saving raw eeg data
            try:
                raw_df = pd.DataFrame(eeg_data[sent_idx]['word_data'][word_idx]['raw_eeg'])
                raw_df.to_csv(sentence_path+'/word_'+str(word_idx)+'_raw.tsv', sep='\t')
            except Exception as e:
                msg = f'Raw EEG extraction error for subject {subject}, sentence {sent_idx}, word {word_idx}:\n{e}\n'
                print(msg)
                #exception_log.write(msg)



if __name__ == '__main__':

    task = 'none'
    while task not in ['NR', 'TSR']:
        task = input('Pick a task to extract ("NR" or "TSR"):\n')

    exception_log = open('exception_log.txt', 'w')

    all_files = get_files()

    # Setting up extraction folder
    extraction_path = EXTRACTED_DATA_PATH+'extracted_data_'+task
    if not os.path.exists(extraction_path):
        os.mkdir(extraction_path)

    # extracting 'basic' data (TODO: find a better name)
    all_items = []
    for file in all_files:
        all_items.append(pd.DataFrame(get_basic_data_from_file(file)))
    basic_data_df = pd.concat(all_items, ignore_index=True)
    basic_data_df.to_csv(extraction_path+"/data.tsv", sep='\t')
    print('Finished extracting basic data.')

    # extracting eeg data
    eeg_path = extraction_path+'/eeg_data'
    if not os.path.exists(eeg_path):
        os.mkdir(eeg_path)
    procs = []
    for file in all_files:
        procs.append(mp.Process(target=process_file, args=(file, eeg_path), name=get_subject(file)))
    index = 0
    while index < len(procs):
        batch_size = min(len(procs)-index, MAX_SIMULTANEOUS_SUBJECTS_TO_PROCESS)
        for i in range(batch_size):
            procs[index+i].start()
            print(f'Processing data for subject {procs[index+i].name}')
        for i in range(batch_size):
            procs[index+i].join()
            print(f'Finished processing data for subject {procs[index+i].name}')
        index+=batch_size
