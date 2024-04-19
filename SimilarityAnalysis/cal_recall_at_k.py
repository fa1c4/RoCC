def calculate_recall_at_k(true_texts, top_k_results):
    true_set = set(true_texts)
    retrieved_set = set([text for text, _ in top_k_results])
    tp = len(true_set & retrieved_set)
    return tp / len(true_set) if len(true_set) > 0 else 0