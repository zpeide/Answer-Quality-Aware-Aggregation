import argparse

from transformers import AutoModelForSequenceClassification, AutoModel, AutoModelForQuestionAnswering, AutoTokenizer
import torch

import random
import numpy  as np
import tqdm
import os
import json
import spacy
import argparse, sys, string
from utils import *


def random_selection(seq_encoding):
    # random selection
    evaluator = get_evaluator(seq_encoding)
    
    rs_scores = None
    num_trials = 3
    f1 = exact_match = total = 0

    for k in range(num_trials):
        eval_results = {}

        for i, _ in enumerate(seq_encoding): #[:100]): 
            all_answer_texts = _[7]
            gt_text = _[6]
            if _[0]:
                gt_idx = len(_[3]) 
                #ground_truth = set(_[5])
                ground_truth = set([gt_idx])
            else:
                ground_truth = set(_[4])
            answer_verify_scores = torch.tensor([s[1] for s in _[3]])
            preds = np.arange(0, len(answer_verify_scores))
            random.shuffle(preds)    

            eval_results[str(i)] = {
                str(j): 1.0/(s+1) for j, s in enumerate(preds)
            }
            

            if not _[0] and len(ground_truth) > 0:
                gt_text = [all_answer_texts[_i] for _i in ground_truth]
                
                exact_match += metric_max_over_ground_truths(exact_match_score, all_answer_texts[preds[0]], gt_text)
                f1 += metric_max_over_ground_truths(f1_score, all_answer_texts[preds[0]], gt_text)


        scores = evaluator.evaluate(eval_results)
        scores = mean_metrics(scores)
        if rs_scores is None:
            rs_scores = scores
        else:
            rs_scores = {_k: rs_scores[_k]+_v for _k, _v in scores.items() }
    rs_scores = {_k:v/num_trials for _k, v in rs_scores.items()}
    em_score = exact_match/len(seq_encoding)/num_trials
    f1_score = f1/len(seq_encoding)/num_trials
    
    return rs_scores, em_score, f1_score


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


def AC(seq_encoding):
    # Only answer verification
    evaluator = get_evaluator(seq_encoding)

    eval_results = {}
    f1 = exact_match = total = 0
    only_answer_verify_scores_preds = []
    for item_idx, _ in enumerate(seq_encoding):     
        gt_idx = len(_[3]) 
        if _[0]:        
            #ground_truth = set(_[5])
            ground_truth = set([gt_idx])
        else:
            ground_truth = set(_[4])
            
        answer_verification_scores = torch.tensor([a[1] for a in _[3] ])
        preds = torch.argsort(answer_verification_scores, descending=True).tolist()
        only_answer_verify_scores_preds.append(preds[0])
        eval_results[str(item_idx)] = {
            str(_): float(answer_verification_scores[_])  for _ in preds
        }    

        all_answer_texts = _[7]
        if not _[0]:
            gt_text = [all_answer_texts[_i] for _i in ground_truth]
            exact_match += metric_max_over_ground_truths(exact_match_score, all_answer_texts[preds[0]], gt_text)
            f1 += metric_max_over_ground_truths(f1_score, all_answer_texts[preds[0]], gt_text)
 
    em_score = exact_match/len(seq_encoding)
    f1 = f1/len(seq_encoding)
    scores = evaluator.evaluate(eval_results)
    scores = mean_metrics(scores)
    
    return scores, em_score, f1


def AC_SMS(seq_encoding, debug_idx=None):
    # SMS with Answer Correctness
    evaluator = get_evaluator(seq_encoding)
    eval_results = {}
    exact_match = f1 = 0.0
    sms_answer_verification = []
    for item_idx, _ in enumerate(seq_encoding):   
        gt_idx = len(_[3])
        answer_verify_scores = torch.tensor([s[1] for s in _[3]])
        _sum = torch.sum(answer_verify_scores)
        if len(_[2]) == 1:
            pred_idx = 0
            preds = [0]
            sms_answer_verification.append(preds[0])
            eval_results[str(item_idx)] = { str(0):1 }          
        else:
            ans_encodings = [_.unsqueeze(0) for _ in _[2]]
            sim_scores = []
            for i in range(len(ans_encodings)):
                temp_encoding = []
                temp_scores = []
                for j in range(len(ans_encodings)):
                    if j != i:
                        temp_encoding.append(ans_encodings[j]) # * answer_verify_scores[j])
                        temp_scores.append(answer_verify_scores[j])
                temp_scores = torch.tensor(temp_scores)
                temp_encoding = torch.cat(temp_encoding)
                sim_scores.append(answer_verify_scores[i] * torch.sum(torch.cosine_similarity(temp_encoding, ans_encodings[i]) * temp_scores ) /torch.sum(temp_scores) )
            sim_scores = torch.tensor(sim_scores)
            pred_sort = torch.argsort(sim_scores, descending=True)
            preds = pred_sort.tolist()
            sms_answer_verification.append(preds[0])
            eval_results[str(item_idx)] = {
                str(_):float(torch.exp(sim_scores[_]))  for _ in preds
            }       
            
            if debug_idx is not None and item_idx == debug_idx:
                sms_sim_scores = sim_scores
                print(f'debug_idx: {debug_idx}, ', sms_sim_scores, ground_truth)     

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


def AC_SMV(seq_encoding):
    # SMV with Answer Correctness
    evaluator = get_evaluator(seq_encoding)

    eval_results = {}
    exact_match = f1 = 0.0
    smv_answer_verification = []
    for item_idx, _ in enumerate(seq_encoding):    # all_question_candidate_answers:    
        gt_idx = len(_[3])
        answer_verify_scores = torch.tensor([s[1] for s in _[3]])
        ans_encodings = torch.cat([a.unsqueeze(0) for a in _[2]])
        __ans_encodings = ans_encodings.clone()
        for i in range(len(answer_verify_scores)):
            __ans_encodings[i, :] = ans_encodings[i, :] # * answer_verify_scores[i]
        candidate = torch.mean(__ans_encodings, dim=0)
        _sum = torch.sum(answer_verify_scores)

        sim_scores = torch.pairwise_distance(ans_encodings, candidate.unsqueeze(0))
        sim_scores = sim_scores / (answer_verify_scores / _sum)
        pred_sort = torch.argsort(sim_scores)

        preds = pred_sort.tolist()
        smv_answer_verification.append(preds[0])
        eval_results[str(item_idx)] = {
            str(_):float(torch.exp(-sim_scores[_]))  for _ in preds
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



def AF(seq_encoding):
    ## answer confidence only
    evaluator = get_evaluator(seq_encoding)

    eval_results = {}
    exact_match = f1 = 0.0
    answer_confidence_preds = []
    for item_idx, _ in enumerate(seq_encoding): 
        if _[0]:
            ground_truth = set([len(_[3])])
        else:
            ground_truth = set(_[4])
        
        answer_confidence_scores = torch.tensor([s for s in _[8]])
        preds = torch.argsort(answer_confidence_scores, descending=True)
        preds = preds.tolist()
        answer_confidence_preds.append(preds[0])

        eval_results[str(item_idx)] = {
            str(j):s for j, s in enumerate(answer_confidence_scores.tolist())
        }            
        # print(ground_truth, preds, _a)
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


def AC_AF(seq_encoding):
    ## answer confidence + answer verification
    evaluator = get_evaluator(seq_encoding)
    eval_results = {}
    exact_match = f1 = 0.0
    ac_af_preds = []
    for item_idx, _ in enumerate(seq_encoding): 
        gt_idx = len(_[3])
        if _[0]:        
            ground_truth = set([gt_idx])
        else:
            ground_truth = set(_[4])
        answer_confidence_scores = torch.tensor([s for s in _[8]])
        answer_verify_scores = torch.tensor([s[1] for s in _[3]])
        scores = (answer_confidence_scores + answer_verify_scores)/2
        preds = torch.argsort(scores, descending=True)
        preds = preds.tolist()
        ac_af_preds.append(preds[0])

        eval_results[str(item_idx)] = {
            str(j):s for j, s in enumerate(scores.tolist())
        }          
        # print(ground_truth, preds, _a)
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


def AF_SMS(seq_encoding):
    # SMS with Answer Confidence
    evaluator = get_evaluator(seq_encoding)
    eval_results = {}
    exact_match = f1 = 0.0
    ac_sms = []
    for item_idx, _ in enumerate(seq_encoding):   
        gt_idx = len(_[3])
        answer_confidence_scores = torch.tensor([s for s in _[8]])
        _sum = torch.sum(answer_confidence_scores)
        if len(_[2]) == 1:
            pred_idx = 0
            preds = [0]
            ac_sms.append(preds[0])
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
                        temp_scores.append(answer_confidence_scores[j])
                temp_scores = torch.tensor(temp_scores)
                temp_encoding = torch.cat(temp_encoding)
                sim_scores.append(answer_confidence_scores[i] * torch.sum(torch.cosine_similarity(temp_encoding, ans_encodings[i]) * temp_scores ) /torch.sum(temp_scores) )
            sim_scores = torch.tensor(sim_scores)
            pred_sort = torch.argsort(sim_scores, descending=True)
            preds = pred_sort.tolist()
            ac_sms.append(preds[0])
            eval_results[str(item_idx)] = {
                str(_):float(torch.exp(sim_scores[_]))  for _ in preds
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
    scores = mean_metrics(scores)
    
    return scores, em_score, f1


def AF_SMV(seq_encoding):
    # SMV with Answer Confidence
    evaluator = get_evaluator(seq_encoding)
    eval_results = {}
    exact_match = f1 = 0.0
    ac_smv = []
    for item_idx, _ in enumerate(seq_encoding):    
        gt_idx = len(_[3])
        answer_confidence_scores = torch.tensor([s for s in _[8]])
        ans_encodings = torch.cat([a.unsqueeze(0) for a in _[2]])
        __ans_encodings = ans_encodings.clone()
        for i in range(len(answer_confidence_scores)):
            __ans_encodings[i, :] = ans_encodings[i, :] 
        candidate = torch.mean(__ans_encodings, dim=0)
        _sum = torch.sum(answer_confidence_scores)

        sim_scores = torch.pairwise_distance(ans_encodings, candidate.unsqueeze(0))
        sim_scores = sim_scores / (answer_confidence_scores / _sum)
        pred_sort = torch.argsort(sim_scores)

        preds = pred_sort.tolist()
        ac_smv.append(preds[0])
        eval_results[str(item_idx)] = {
            str(_):float(torch.exp(-sim_scores[_]))  for _ in preds
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




def main():
    dev_ds = args.dev_ds # "nq_dev.json"
    dev_ds = json.load(open(dev_ds))['data']
    dataset_subset = args.dataset_subset
    model_name = args.model_name 


    model_name_in_path = model_name.replace('/', '-')
    all_encodings = torch.load(f"{dataset_subset}_{model_name_in_path}_encoding.pt")
    answer_free_seq_encoding = torch.load(f"{dataset_subset}_all_answer_candidate_embeddings_{model_name_in_path}.pt")
    all_question_candidate_answers_with_null_candidate = torch.load(f"{dataset_subset}_all_question_candidate_answers_qa_no_noanswer_add_null_{model_name_in_path}.pt")  

    rs_scores, em_rs, f1_rs = random_selection(answer_free_seq_encoding)
    print_results("Context Free SMS", rs_scores, em_rs, f1_rs)

    ac_scores, em_ac, f1_ac = AC(all_question_candidate_answers_with_null_candidate)    
    print_results("AC", ac_scores, em_ac, f1_ac)

    af_scores, em_af, f1_af = AF(all_question_candidate_answers_with_null_candidate)    
    print_results("AF", af_scores, em_af, f1_af)

    acaf_scores, em_acaf, f1_acaf = AC_AF(all_question_candidate_answers_with_null_candidate)    
    print_results("AF", acaf_scores, em_acaf, f1_acaf)

    sms_scores, em_socre_sms, f1_score_sms = SMS(answer_free_seq_encoding)
    print_results("Context Free SMS", sms_scores, em_socre_sms, f1_score_sms)
    smv_scores, em_socre_smv, f1_score_smv = SMV(answer_free_seq_encoding)
    print_results("Context Free SMV", smv_scores, em_socre_smv, f1_score_smv)

    sms_scores, em_socre_sms, f1_score_sms = SMS(all_question_candidate_answers_with_null_candidate)
    print_results("Context Aware SMS", sms_scores, em_socre_sms, f1_score_sms)
    smv_scores, em_socre_smv, f1_score_smv = SMV(all_question_candidate_answers_with_null_candidate)
    print_results("Context Aware SMV", smv_scores, em_socre_smv, f1_score_smv)

    acsms_scores, em_acsms, f1_acsms = AC_SMS(all_question_candidate_answers_with_null_candidate)
    print_results("AC SMS", acsms_scores, em_acsms, f1_acsms)
    acsmv_scores, em_socre_acsmv, f1_score_acsmv = AC_SMV(all_question_candidate_answers_with_null_candidate)
    print_results("AC SMV", acsmv_scores, em_socre_acsmv, f1_score_acsmv)

    afsms_scores, em_afsms, f1_afsms = AF_SMS(all_question_candidate_answers_with_null_candidate)
    print_results("AC SMS", afsms_scores, em_afsms, f1_afsms)
    afsmv_scores, em_socre_afsmv, f1_score_afsmv = AF_SMV(all_question_candidate_answers_with_null_candidate)
    print_results("AC SMV", afsmv_scores, em_socre_afsmv, f1_score_afsmv)

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
