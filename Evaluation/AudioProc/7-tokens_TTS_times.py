import json 
import nltk


with open('./jsons/TTS_times.json', 'r', encoding='utf-8') as f:
    tts_times = json.loads(f.read())

time_total = 0.0
tokens_total = 0
for key, value in tts_times.items():
    tokened_pred = nltk.word_tokenize(value[0])
    tokens_count = len(tokened_pred)
    tokens_total += tokens_count
    time_total += value[1]

print('----- time comsumed for TTS: {}s -----'.format(time_total))
print('----- average tokens of TTS per second: {}s -----'.format(tokens_total / time_total))
print('----- total tokens of TTS: {} -----'.format(tokens_total))