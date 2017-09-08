import librosa
import pickle

import os
from itertools import groupby

SRT_DIR = '../data/srts/'
AUDIO_DIR = '../data/audio-files/'

list_srts = os.listdir(SRT_DIR)
list_audio_files = os.listdir(AUDIO_DIR)

def np_index(time_string,sr):
    hr=int(time_string[0:2])
    mins=int(time_string[3:5])
    sec=int(time_string[6:8])
    milli_secs=int(time_string[9:])  
    return int((hr*3600+mins*60+sec+milli_secs*0.001)*sr)

total_sentences_loaded = 0
total_files_loaded = 0
# list_srts = ['1.srt']
skip_talks = ['1738', '9', '1', '2178']
training_data = []

# print 'Total number of srts:', len(list_srts)

for srt_file in list_srts:

    talk_id = srt_file[:-4]
    srt_filename = SRT_DIR + srt_file
    if talk_id in skip_talks:
	continue

    audio_file = talk_id + '.wav'
    if audio_file not in list_audio_files:
        continue
    audio_filename = AUDIO_DIR + audio_file

    print 'Loading talk ID:', talk_id

    srt_f=open(srt_filename,'r')
    data_array=[]

    for line in srt_f:
        data_array.append(line)

    a = groupby(data_array, lambda x:x=='\n')
    data_list = [list(group) for k, group in a if not k]
    stripped_data_list = []
    for group in data_list:
        stripped_data_list.append(map(str.strip, group))
    time_array = [x[1] for x in stripped_data_list]
    sentence_array = [' '.join(x[2:]) for x in stripped_data_list]

    start_timestamp=[]
    end_timestamp=[]

    # print 'num sentences:', len(sentence_array)
    # print 'Calculating timestamps...'
    for i in range(len(time_array)):
        time_array[i]=time_array[i][:-1]
        start_timestamp.append(time_array[i].split(' --> ')[0])
        end_timestamp.append(time_array[i].split(' --> ')[1])

    y,sr=librosa.load(audio_filename)

    # print 'Splicing audio...'
    np_sliced_list=[] 
    #List of np arrays
    for j in range(len(time_array)):
        time_slice=y[np_index(start_timestamp[j],sr):np_index(end_timestamp[j],sr)]
        np_sliced_list.append(time_slice)


    # print 'Getting audio features...'
    mfcc_sliced_list=[]
    #List of mfcc np arrays
    for j in range(len(np_sliced_list)):
        mfcc_slice=librosa.feature.mfcc(np_sliced_list[j])
        mfcc_sliced_list.append(mfcc_slice)

    # print 'mffc_sliced len:', len(mfcc_sliced_list)
    # print 'num sentences:', len(sentence_array)
    if len(mfcc_sliced_list) != len(sentence_array):
        print 'Conflict!'
        continue

    total_files_loaded += 1
    total_sentences_loaded += len(sentence_array)

    for i in range(len(sentence_array)):
        training_data.append((mfcc_sliced_list[i], sentence_array[i]))

    print total_files_loaded, ' files loaded'

fp = open('training_data.pkl', 'wb')
pickle.dump(training_data, fp)

print 'Total files loaded:', total_files_loaded
print 'Total sentences loaded:', total_sentences_loaded

