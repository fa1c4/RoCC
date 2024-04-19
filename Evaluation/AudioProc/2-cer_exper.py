'''
Ths experiment 2, computes the cer of model of 'microsoft/speecht5_asr'
two datasets including: 
(1) hf-internal-testing/librispeech_asr_dummy
(2) hf-internal-testing/librispeech_asr_demo
'''

import json
from evaluate import load

cer = load('cer')
wer = load('wer')
# ser = load('ser')
def compute_ser(predictions:list, references:list):
    if len(predictions) != len(references):
        print('different length of two lists between predictions({}) and references({})'
              .format(len(predictions), len(references)))
        exit(1)
    
    length = len(predictions)
    matched_cnt = 0
    for i in range(length):
        if predictions[i] == references[i]:
            matched_cnt += 1
    
    return 1.0 - matched_cnt / length 


with open('./jsons\\librispeech_asr_demo_results.json', 'r', encoding='utf-8') as f:
# with open('J:\Covert_Channel\speechT5\jsons\librispeech_asr_demo_results.json', 'r', encoding='utf-8') as f:
    json_data1 = f.read()
    json_data1 = json.loads(json_data1)

with open('./jsons/librispeech_asr_dummy_results.json', 'r', encoding='utf-8') as f:
    json_data2 = json.loads(f.read())


# print(json_data)
preds = []
oris = []
for key, value in json_data1.items():
    # pred = value[0].split(' ')
    # ori = value[1].split(' ')
    preds.append(value[0])
    oris.append(value[1])
    # print(preds, oris)

for key, value in json_data2.items():
    preds.append(value[0])
    oris.append(value[1])

ser_scores = {}
cer_scores = {}
for i in range(0, len(preds), 20):
    if i + 20 > len(preds):
        next_step = len(preds)
    else: 
        next_step = i + 20

    pred_temp = preds[0: next_step]
    ori_temp = oris[0: next_step]
    # print(len(pred_temp), pred_temp)
    cer_score = cer.compute(predictions=pred_temp, references=ori_temp)
    cer_scores[next_step] = cer_score
    print('----- {} sentences cer score: {} -----'.format(next_step, cer_score))

    ser_score = compute_ser(predictions=pred_temp, references=ori_temp)
    ser_scores[next_step] = ser_score
    print('----- {} sentences ser score: {} -----'.format(next_step, ser_score))


with open('./jsons/asr_cer_speechT5.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(cer_scores))

print('------ cer computation completed ------')

with open('./jsons/asr_ser_speechT5.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(ser_scores))

print('------ ser computation completed ------')