from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
import time
import json
import os


generator = pipeline(task="automatic-speech-recognition", model="microsoft/speecht5_asr")

dir_path = './wavs'
audio_recogs = {}
debug_flag = False
test_cnt = 0

# traversal all files in the target directory
def getfiles():
    filenames = os.listdir(dir_path)
    print(filenames)
    return filenames

# get the origin texts from file TTS_times.json
def get_ori_texts():
    with open('./jsons/TTS_times.json', 'r', encoding='utf-8') as f:
        json_data = json.loads(f.read())

    ret_ori_texts = []
    for k, v in json_data.items():
        ret_ori_texts.append(v[0])
    
    return ret_ori_texts


if __name__ == '__main__':
    filenames = getfiles()
    ori_texts = get_ori_texts()

    for filename in filenames:
        # debug if debug_flag == True 
        # with test_cnt < number
        if 'example' in filename:
            continue

        test_cnt += 1
        if debug_flag:
            if test_cnt > 10:
                break
        
        index = int(filename.strip('.wav').split('_')[-1])
        # print(int(index))
        # input('press any key to continue...')
        audio_file = dir_path + '/' + filename
        test_audio_file, samplerate = sf.read(audio_file, start=0, stop=None)

        start_time = time.time()
        test_wav_gen = generator(test_audio_file)
        end_time = time.time()

        time_comsumed = end_time - start_time
        print('time comsumed: {} s'.format(time_comsumed))
        print(test_wav_gen)

        ori_text = ori_texts[index]
        print('origin text is: {}'.format(ori_text))
        audio_recogs[index] = [test_wav_gen['text'], ori_text, time_comsumed]


    # with open('./librispeech_asr_demo_results.json', 'w', encoding='utf-8') as f:
    with open('./jsons/stego_texts_results.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(audio_recogs, indent=4))

    print('----- asr completed -----')