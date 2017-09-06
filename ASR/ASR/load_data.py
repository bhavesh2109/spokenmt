import librosa

import os

SRT_DIR = '../data/srts/'
AUDIO_DIR = '../data/audio-files'

list_srts = os.listdir(SRT_DIR)
list_audio_files = os.listdir(AUDIO_DIR)

for srt_file in list_srts:

    srt_id = srt_file[:-4]
    srt_filename = SRT_DIR + srt_file

    audio_file = srt_id + '.wav'
    if audio_file not in list_audio_files:
        continue

    audio_filename = AUDIO_DIR + audio_file

    
