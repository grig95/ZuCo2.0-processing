import os
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import data_loading_helpers as dh

def get_subject(filename):
    fis = os.path.basename(filename)
    return fis.split("sults")[1].split("_")[0]

def get_task(filename):
    fis = os.path.basename(filename)
    return fis.split("_")[1].replace(".mat", '')

def get_files(task = 'NR'):
    if task == 'NR' or task == 1:
        task = 'NR'
        rootdir = 'task1 - NR/Matlab files/'
    elif task == 'TSR' or task == 2:
        task = 'TSR'
        rootdir = 'task2 - TSR/Matlab files/'
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


def get_data_from_file(file, columns=['content', 'word_idx', 'FFD', 'GD', 'GPT', 'TRT', 'nFix', 'reading_order']):
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
            print(sent_idx)
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



if __name__ == '__main__':
    all_files = get_files()
    print(all_files)
    #print('number of sentences: ', len(rawData))
    
    #columns=['content', 'FFD', 'GD', 'GPT', 'TRT', 'nFix', 'reading_order']
    all_items = []
    for file in all_files:
        all_items.append(pd.DataFrame(get_data_from_file(file)))


    df = pd.concat(all_items, ignore_index=True)

    df.to_csv("data.tsv", sep='\t')
