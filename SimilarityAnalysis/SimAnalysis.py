import json
import numpy as np
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel
import torch
from Levenshtein import distance as levenshtein_distance
from cal_recall_at_k import calculate_recall_at_k

# loading BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def preprocess(text):
    # normalize text
    return text.lower().strip()

def get_embedding(text):
    # get BERT embedding
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

def calculate_similarity(text1, text2, alpha=0.5):
    # get similarity between texts
    text1 = preprocess(text1)
    text2 = preprocess(text2)

    # compute levenshtein_distance
    edit_dist = levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    normalized_edit_sim = 1 - (edit_dist / max_len)

    # compute cosine similarity
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    cosine_sim = cosine(emb1, emb2)

    # combine the two similarities and use a weighted alpha
    return (1-alpha) * normalized_edit_sim + alpha * cosine_sim

def find_top_similar(target_text, json_data, top_n=10):
    """get the top_n most similar texts to the target_text from the json_data"""
    similarities = []
    for entry in json_data:
        sim = calculate_similarity(target_text, entry["text"], alpha=0.5)
        similarities.append((entry["text"], sim))

    # rank as per similarity and return top_n texts
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_similarities[:top_n]

# get the texts data
with open('data.json', 'r') as file:
    data = json.load(file)

# test the target text
target_text = "Example target text to compare."

# get the top n similar texts
top_texts = find_top_similar(target_text, data, top_n=10)
for text, sim in top_texts:
    print(f"Text: {text}, Similarity: {sim}")

# calculate the recall at k
recall_k = calculate_recall_at_k(target_text, top_texts)
print('Recall at k:', recall_k)