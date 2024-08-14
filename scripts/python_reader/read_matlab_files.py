import os
import numpy as np
import h5py
import data_loading_helpers as dh

task = "TSR"

rootdir = "/Volumes/methlab/NLP/Ce_ETH/2019/FirstLevel_V2/"


sentences = {}

for file in os.listdir(rootdir):
    if file.endswith(task+".mat"):
        print(file)

        file_name = rootdir + file
        subject = file_name.split("ts")[1].split("_")[0]

        # exclude YMH due to incomplete data because of dyslexia
        if subject != 'YMH':

            f = h5py.File(file_name)
            sentence_data = f['sentenceData']
            rawData = sentence_data['rawData']
            contentData = sentence_data['content']
            omissionR = sentence_data['omissionRate']
            wordData = sentence_data['word']

            # number of sentences:
            # print(len(rawData))

            for idx in range(len(rawData)):
                obj_reference_content = contentData[idx][0]
                sent = dh.load_matlab_string(f[obj_reference_content])

                # get omission rate
                obj_reference_omr = omissionR[idx][0]
                omr = np.array(f[obj_reference_omr])
                print(omr)

                # get word level data
                word_data = dh.extract_word_level_data(f, f[wordData[idx][0]])

                # number of tokens
                # print(len(word_data))

                for widx in range(len(word_data)):

                    # get first fixation duration (FFD)
                    print(word_data[widx]['FFD'])

                    # get aggregated EEG alpha features
                    print(word_data[widx]['ALPHA_EEG'])


