import argparse

from transformers import BertForSequenceClassification, BertTokenizerFast, AutoModelForSequenceClassification, AutoModel, AutoModelForQuestionAnswering, AutoTokenizer

import torch

import logging
import random
import numpy  as np
import tqdm
import os
import json
import spacy

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import re, argparse, sys, string
from collections import Counter
import pytrec_eval



device_no   =   0
qa_device   =   0


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


def SMS(seq_encoding):
    # Context-Free SMS
    evaluator = get_evaluator(seq_encoding)
    eval_results = {}

    f1 = exact_match = total = 0
    context_free_sms = []
    for item_idx, _ in enumerate(seq_encoding):    
        gt_idx = len(_[2])
        if len(_[2]) == 1:
            preds = [0]
            context_free_sms.append(preds[0])
            eval_results[str(item_idx)] = {
                str(0):1
            }    
        else:
            ans_encodings = [_.unsqueeze(0) for _ in _[2]]
            sim_scores = []
            for i in range(len(ans_encodings)):
                temp_encoding = []
                for j in range(len(ans_encodings)):
                    if j != i:
                        temp_encoding.append(ans_encodings[j])
                temp_encoding = torch.cat(temp_encoding)

                sim_scores.append(torch.mean(torch.pairwise_distance(ans_encodings[i], temp_encoding)))
            pred_sort = torch.argsort(torch.tensor(sim_scores)) #, descending=True)
            
            preds = pred_sort.tolist()
            context_free_sms.append(preds[0])
            eval_results[str(item_idx)] = {
                str(_):float(torch.exp(-sim_scores[_]))  for _ in preds
            }    
        target_idxs = _[4]
        no_ans_idxs = _[5]
        if _[0]:
            ground_truth = set([gt_idx])
        else:        
            ground_truth = set(_[4])


        all_answer_texts = _[7]
        if not _[0] and len(ground_truth) > 0:
            gt_text = [all_answer_texts[_i] for _i in ground_truth]
            
            exact_match += metric_max_over_ground_truths(exact_match_score, all_answer_texts[preds[0]], gt_text)
            f1 += metric_max_over_ground_truths(f1_score, all_answer_texts[preds[0]], gt_text)

    em_score = exact_match/len(seq_encoding)
    f1 = f1/len(seq_encoding)
    scores = evaluator.evaluate(eval_results)
    sms_scores = mean_metrics(scores)
    
    return sms_scores, em_score, f1


def SMV(seq_encoding):
    # Context-Free SMV
    evaluator = get_evaluator(seq_encoding)
    eval_results = {}
    exact_match = f1 = 0.0
    context_free_smv = []
    for item_idx, _ in enumerate(seq_encoding):   
        gt_idx = len(_[2]) 
        ans_encodings = torch.cat([_.unsqueeze(0) for _ in _[2]])
        candidate = torch.mean(torch.cat([_.unsqueeze(0) for _ in ans_encodings]), dim=0)
        # sim_scores = torch.cosine_similarity(ans_encodings, candidate.unsqueeze(0))
        # pred_sort = torch.argsort(sim_scores, descending=True)
        sim_scores = torch.pairwise_distance(ans_encodings, candidate.unsqueeze(0))    
        pred_sort = torch.argsort(sim_scores)
        preds = pred_sort.tolist()
        context_free_smv.append(preds[0])
        target_idxs = _[4]

        eval_results[str(item_idx)] = {
            str(_):float(torch.exp(-sim_scores[_]))  for _ in preds
            # str(j):float(answer_verification_scores[j])  for _, j in enumerate(preds)
        }  
        if _[0]:
            ground_truth = set([gt_idx])
        else:
            ground_truth = set(_[4])

        all_answer_texts = _[7]
        if not _[0] and len(ground_truth) > 0:
            gt_text = [all_answer_texts[_i] for _i in ground_truth]
            
            exact_match += metric_max_over_ground_truths(exact_match_score, all_answer_texts[preds[0]], gt_text)
            f1 += metric_max_over_ground_truths(f1_score, all_answer_texts[preds[0]], gt_text)

    em_score = exact_match/len(seq_encoding)
    f1 = f1/len(seq_encoding)
    scores = evaluator.evaluate(eval_results)
    sms_scores = mean_metrics(scores)
    
    return sms_scores, em_score, f1
    

def ACAF_SMS(seq_encoding):
    # SMS with Answer Confidence + Answer Verification
    evaluator = get_evaluator(seq_encoding)
    eval_results = {}
    exact_match = f1 = 0.0
    ac_af_sms = []
    for item_idx, _ in enumerate(seq_encoding):     
        gt_idx = len(_[3])
        answer_confidence_scores = torch.tensor([s for s in _[8]])
        answer_verify_scores = torch.tensor([s[1] for s in _[3]])
        scores = answer_confidence_scores + answer_verify_scores    
        _sum = torch.sum(scores)
        if len(_[2]) == 1:
            pred_idx = 0
            preds = [0]
            ac_af_sms.append(preds[0])
            eval_results[str(item_idx)] = { str(0):1 }          
        else:
            ans_encodings = [_.unsqueeze(0) for _ in _[2]]
            sim_scores = []
            for i in range(len(ans_encodings)):
                temp_encoding = []
                temp_scores = []
                for j in range(len(ans_encodings)):
                    if j != i:
                        temp_encoding.append(ans_encodings[j]) 
                        temp_scores.append(scores[j])
                temp_scores = torch.tensor(temp_scores)
                temp_encoding = torch.cat(temp_encoding)
                sim_scores.append(scores[i] * torch.sum(torch.cosine_similarity(temp_encoding, ans_encodings[i]) * temp_scores ) /torch.sum(temp_scores) )
            sim_scores = torch.tensor(sim_scores)
            pred_sort = torch.argsort(sim_scores, descending=True)
            preds = pred_sort.tolist()
            ac_af_sms.append(preds[0])
            eval_results[str(item_idx)] = {
                str(_):float(torch.exp(sim_scores[_]))  for _ in preds
            }           

        if _[0]:        
            ground_truth = set([gt_idx])    
        else:
            ground_truth = set(_[4])
        
        all_answer_texts = _[7]
        if not _[0] and len(ground_truth) > 0:
            gt_text = [all_answer_texts[_i] for _i in ground_truth]
            
            exact_match += metric_max_over_ground_truths(exact_match_score, all_answer_texts[preds[0]], gt_text)
            f1 += metric_max_over_ground_truths(f1_score, all_answer_texts[preds[0]], gt_text)

    em_score = exact_match/len(seq_encoding)
    f1 = f1/len(seq_encoding)
    scores = evaluator.evaluate(eval_results)
    scores = mean_metrics(scores)
    
    return scores, em_score, f1


def ACAF_SMV(seq_encoding):
    # SMV with Answer Confidence + Answer Verification
    evaluator = get_evaluator(seq_encoding)
    eval_results = {}
    exact_match = f1 = 0.0
    acaf_smv = []
    for item_idx, _ in enumerate(seq_encoding):    # all_question_candidate_answers:    
        gt_idx = len(_[3])
        answer_confidence_scores = torch.tensor([s for s in _[8]])
        answer_verify_scores = torch.tensor([s[1] for s in _[3]])
        scores = answer_confidence_scores + answer_verify_scores      
        ans_encodings = torch.cat([a.unsqueeze(0) for a in _[2]])
        __ans_encodings = ans_encodings.clone()
        for i in range(len(scores)):
            __ans_encodings[i, :] = ans_encodings[i, :] 
        candidate = torch.mean(__ans_encodings, dim=0)
        _sum = torch.sum(scores)
        
        sim_scores = torch.pairwise_distance(ans_encodings, candidate.unsqueeze(0))
        sim_scores = sim_scores / (scores / _sum)
        pred_sort = torch.argsort(sim_scores)


        preds = pred_sort.tolist()
        acaf_smv.append(preds[0])

        if _[0]:        
            ground_truth = set([gt_idx])    
        else:
            ground_truth = set(_[4])

        eval_results[str(item_idx)] = {
            str(_):float(torch.exp(-sim_scores[_]))  for _ in preds
        }   
        all_answer_texts = _[7]
        if not _[0] and len(ground_truth) > 0:
            gt_text = [all_answer_texts[_i] for _i in ground_truth]
            
            exact_match += metric_max_over_ground_truths(exact_match_score, all_answer_texts[preds[0]], gt_text)
            f1 += metric_max_over_ground_truths(f1_score, all_answer_texts[preds[0]], gt_text)
    em_score = exact_match/len(seq_encoding)
    f1 = f1/len(seq_encoding)
    scores = evaluator.evaluate(eval_results)
    scores = mean_metrics(scores)
    
    return scores, em_score, f1



def print_results(method, scores, em, f1):
    print(f"Method: {method}")
    print(scores)
    print(f'exact match: {em}; f1: {f1}')


def main():
    dev_ds = args.dev_ds # "nq_dev.json"
    dev_ds = json.load(open(dev_ds))['data']
    dataset_subset = args.dataset_subset
    model_name = args.model_name 


    model_name_in_path = model_name.replace('/', '-')
    all_encodings = torch.load(f"{dataset_subset}_{model_name_in_path}_encoding.pt")
    answer_free_seq_encoding = torch.load(f"{dataset_subset}_all_answer_candidate_embeddings_{model_name_in_path}.pt")
    all_question_candidate_answers_with_null_candidate = torch.load(f"{dataset_subset}_all_question_candidate_answers_qa_no_noanswer_add_null_{model_name_in_path}.pt")  

    sms_scores, em_socre_sms, f1_score_sms = SMS(answer_free_seq_encoding)
    print_results("SMS", sms_scores, em_socre_sms, f1_score_sms)
    smv_scores, em_socre_smv, f1_score_smv = SMV(answer_free_seq_encoding)
    print_results("SMV", smv_scores, em_socre_smv, f1_score_smv)
    acaf_sms_scores, em_acaf_sms, f1_acaf_sms = ACAF_SMS(all_question_candidate_answers_with_null_candidate)
    print_results("ACAF-SMS", acaf_sms_scores, em_acaf_sms, f1_acaf_sms)
    acaf_smv_scores, em_acaf_smv, f1_acaf_smv = ACAF_SMV(all_question_candidate_answers_with_null_candidate)
    print_results("ACAF-SMV", acaf_smv_scores, em_acaf_smv, f1_acaf_smv)


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('--model_name', help='Encoder Name')
    params.add_argument('--dataset_subset', choices=['train', 'dev', 'test'], default='dev', help='subset name')
    params.add_argument('--dev_ds', type=str, default="nq_dev.json", help='path to the squad format data file.')
    
    args = params.parse_args()

    main(args)
