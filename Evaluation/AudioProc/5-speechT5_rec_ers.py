'''
this lab intends to compute the CER and WER of the stego texts
the WER is the same as SER (sentence error rate)
'''

import json
from evaluate import load
import nltk


cer = load("cer")
wer = load('wer')

def compute_ser(predictions:list, references:list):
    '''
    compute the sentence error rate
    '''
    assert len(predictions) == len(references)
    total = len(predictions)
    error = 0
    for i in range(total):
        if predictions[i] != references[i]:
            error += 1
    
    return error / total


with open('./jsons/stego_texts_results.json', 'r', encoding='utf-8') as f:
    asr_data = json.loads(f.read())

preds = []
oris = []
time_comsumed = 0.0
tokens_total = 0
for index, value in asr_data.items():
    pred = value[0]
    ori = value[1]
    if pred == '': continue
    if ori == '': continue

    tokened_pred = nltk.word_tokenize(pred)
    tokens_total += len(tokened_pred)

    preds.append(pred)
    oris.append(ori)
    time_comsumed += value[2]

cer_score = cer.compute(predictions=preds, references=oris)
wer_score = compute_ser(predictions=preds, references=oris)
print('----- CER score: {} -----'.format(cer_score))
print('----- SER score: {} -----'.format(wer_score))

print('----- time comsumed for Recognition: {}s -----'.format(time_comsumed))
print('----- average time of Recognition per token: {}s -----'.format(time_comsumed / tokens_total))