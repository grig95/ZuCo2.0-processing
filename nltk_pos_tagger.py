# This script reads the data.tsv file and generates a separate tsv file tagging the words in each sentence with their part-of-speech, according to nltk.pos_tag()

import pandas as pd
import nltk
import os

from commons import get_input_from_list
from reader import EXTRACTED_DATA_PATH
from reader import get_subjects_list

if __name__ == '__main__':
    task = get_input_from_list(['NR', 'TSR'], f'Generate for which task? ("NR" or "TSR"):\n')
    data_df = pd.read_csv(f'{EXTRACTED_DATA_PATH}extracted_data_{task}/data.tsv', sep='\t', keep_default_na=False) # 'None' is a keyword interpreted as NaN (and actual missing values are marked with '')
    reference_subject = get_subjects_list()[0] # the sentences are identical for all subjects, so only one needs to be processed
    subject_df = data_df[ data_df['subject']==reference_subject ]
    sent_dfs = []
    for sentence_id in pd.unique(subject_df['sentence_id']):
        sentence_df = subject_df[ subject_df['sentence_id']==sentence_id ] [ ['task', 'sentence', 'sentence_id', 'content', 'word_idx'] ]
        tagged_words = nltk.pos_tag(sentence_df['content'].to_list())
        sentence_df.insert(4, 'part_of_speech', [t[1] for t in tagged_words])
        sent_dfs.append(sentence_df)
    pos_tagged_df = pd.concat(sent_dfs, ignore_index=True)
    if not os.path.exists(f'{EXTRACTED_DATA_PATH}extracted_data_{task}/pos_tags'):
        os.mkdir(f'{EXTRACTED_DATA_PATH}extracted_data_{task}/pos_tags')
    pos_tagged_df.to_csv(f'{EXTRACTED_DATA_PATH}extracted_data_{task}/pos_tags/nltk.tsv', sep='\t', index=False)