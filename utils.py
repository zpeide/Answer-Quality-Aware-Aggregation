import pytrec_eval
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
import string
from collections import Counter
import numpy as np


def evaluate_performance(ground_truth, prediction):
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, prediction, average='binary')
    acc = accuracy_score(ground_truth, prediction)
    print(precision, 'recall:', recall, 'f1:', f1, 'acc:', acc)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    # print(ground_truths)
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_evaluator(all_question_candidate_answers_with_null_candidate):
    all_ground_truth = {}
    for i, _ in enumerate( all_question_candidate_answers_with_null_candidate) : 
        all_ground_truth[str(i)] = {}
        if _[0]:
            all_ground_truth[str(i)] = {
                str(len(answer_idx)): 1,
            }
        else:
            answer_idx = _[4]
            for j in range(len(_[2])):
                if j in answer_idx:
                    all_ground_truth[str(i)][str(j)] = 1
                else:
                    all_ground_truth[str(i)][str(j)] = 0
    evaluator = pytrec_eval.RelevanceEvaluator(all_ground_truth, {'map', 'map_cut_1,2', 'ndcg', 'ndcg_cut_1,2',  'P_1', 'P_2', 'P_3', 'recall', 'recall_1', 'recall_3', 'recall_5' })    
    return evaluator

def mean_metrics(scores):
    vs = list(scores.values())
    metrics = vs[0].keys()
    print(metrics)
    avg_metrics = {}
    for k in metrics:
        d = [_[k] for _ in vs]
        avg_metrics[k] = np.mean(d)
    return avg_metrics


def print_results(method, scores, em, f1):
    print(f"Method: {method}")
    print(scores)
    print(f'exact match: {em}; f1: {f1}')

