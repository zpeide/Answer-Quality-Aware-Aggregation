import argparse

from transformers import BertTokenizerFast, AutoModelForSequenceClassification, AutoModel, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import torch

from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler

import logging
import random
import numpy  as np
import os
import tqdm
# import wandb
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import hashlib
from nltk import sent_tokenize
import spacy


device_no   =   0
qa_device   =   0


def get_all_encodings(file_path, model_name, dev_ds):
    tokenizer =  AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if os.path.exists(file_path):
        all_encodings = torch.load(file_path)
    else:
        model = AutoModel.from_pretrained(model_name).eval().to(device_no)
        keys = ['input_ids', 'attention_mask'] #'token_type_ids', 
        all_encodings = {}
        for i, item in enumerate(tqdm.tqdm(dev_ds)):
            for para in item['paragraphs']:
                src = para['context']
                storyid = hashlib.md5(src.encode('utf-8')).hexdigest()
                if storyid in all_encodings:
                    continue
                # print(src)
                tokenized_src = tokenizer(
                    src,                
                    max_length=512,                                        
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,    
                )
                encodings = []
                for _o, offsets in enumerate(tokenized_src['offset_mapping']):
                    data = {k: torch.tensor(tokenized_src[k][_o]).unsqueeze(0).to(device_no) for k in keys}
                    bert_encodings = model(**data)
                    bert_encodings = bert_encodings[0].cpu().detach()
                    encodings.append((offsets, bert_encodings))
                all_encodings[storyid] = encodings
        torch.save(all_encodings, file_path)
    return all_encodings


def get_all_candidate_answers_as_free_sequence(file_path, model_name, all_encodings, device_no, dev_ds, include_no_answer=True, re_calc=False, add_null_candidate=False):
    with torch.no_grad():
        tokenizer =  AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if os.path.exists(file_path):
            all_question_candidate_answers = torch.load(file_path)
        else:
            model = AutoModel.from_pretrained(model_name).eval().to(device_no)
            all_question_candidate_answers = []
            for i, item in enumerate(tqdm.tqdm(dev_ds)):
                for para in item['paragraphs']:
                    src = para['context']                    
                    storyid = hashlib.md5(src.encode('utf-8')).hexdigest()
                    sents = sent_tokenize(src)
                    
                    encodings = all_encodings[storyid]
                    offsetmaps = [_[0] for _ in encodings]
                    encodings = [_[1].squeeze() for _ in encodings]

                    for j, question in enumerate(para['qas']):
                        answers = question['answers']
                        consensus = None
                        if len(answers) > 2:
                            answer_bounds = [(_['answer_start'], _['answer_start']+len(_['text'])) for _ in answers]
                            answer_counts = {f'{la[0]}-{la[1]}':answer_bounds.count(la) for la in answer_bounds}
                            answer_counts = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
                            # we only keep answers with disagreement for testing.
                            if len(answer_counts) > 1 and answer_counts[0][1] > answer_counts[1][1]:                                
                                consensus = {'s': int(answer_counts[0][0].split('-')[0]), 'e': int(answer_counts[0][0].split('-')[1])}
                        
                        if consensus is None:
                            continue
                    
                        target_ans_idx = -1
                        no_ans = False
                        ans_encodings = []
                        all_ans_scores = []
                        no_ans_idx = []
                        all_ans_text = []
                        ans_idx = []

                        consensus_ans_text = src[consensus['s']:consensus['e']].strip()
                        # aggregation after validation. remove duplicate answers.            
                        encoded_answers = set()
                        for answer in answers:
                            ans = answer
                            assert src[answer['answer_start']:answer['answer_start']+len(answer['text'])] == answer['text']
                            ans['s'] = ans['answer_start']
                            ans['e'] = ans['answer_start'] + len(ans['text'])
                            ans_key = f"{ans['s']}-{ans['e']}"
                            if ans_key in encoded_answers:
                                continue
                            else:
                                encoded_answers.add(ans_key)
                            ans_text = src[ans['s']:ans['e']].strip().lower()
                            ans_token_encodings = []
                            tokenized_ans = tokenizer(ans_text, return_tensors='pt').to(device_no)
                            ans_token_encodings = model(**tokenized_ans)
                            ans_token_encodings = ans_token_encodings[0].cpu().detach()

                            ans_pooled_encoding = torch.mean(ans_token_encodings[0,:,:], dim=0)
                            
                            if not no_ans and ans['s'] <= consensus['e'] and ans['e'] >= consensus['s']:
                                _x = len(src[consensus['s']:consensus['e']].split())
                                if len(src[ans['s']:ans['e']].split()) in range(_x-1, _x+1):
                                    ans_idx.append(len(ans_encodings))
                            ans_encodings.append(ans_pooled_encoding)

                            all_ans_text.append(ans_text)
                            
                        all_question_candidate_answers.append([no_ans, None, ans_encodings, None, ans_idx, no_ans_idx, consensus_ans_text, all_ans_text, None, None, None])   

            torch.save(all_question_candidate_answers, file_path)    
    return all_question_candidate_answers


def ans_confidence(qa_tokenizer, offset_mapping, t, ans, out):
    cls_probs = []
    ans_probs = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = t['input_ids'][i]
        cls_index = input_ids.tolist().index(qa_tokenizer.cls_token_id)
        # sequence corresponding to the example ( to know what is the context and what is the question).
        sequence_ids = t.sequence_ids(i)

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(sequence_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        start_logits = out.start_logits[i, :].softmax(-1).cpu().detach()
        end_logits = out.end_logits[i, :].softmax(-1).cpu().detach()
        cls_prob = torch.sqrt(start_logits[cls_index]*end_logits[cls_index])
        cls_probs.append(cls_prob)
        ans_start_idx = token_start_index
        ans_end_idx = token_end_index
        if 's' not in ans or 'e' not in ans:
            ans_start_idx = cls_index
            ans_end_idx = cls_index
            ans_prob = cls_prob            
        else:
            start_char = ans['s']
            end_char = ans['e']
            # print(i, '---', ans, token_start_index, token_end_index, len(offsets), offsets[token_start_index][0], offsets[token_end_index][1])
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            # if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            #     continue            
                
            if not (offsets[token_start_index][0] < end_char and offsets[token_end_index][1] > start_char):
                continue
            else:
                
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] < start_char:
                    token_start_index += 1
                # print(token_start_index, offsets[token_start_index][0], start_char)
                ans_start_idx = token_start_index - 1
                while offsets[token_end_index][1] > end_char and token_end_index > token_start_index:
                    token_end_index -= 1
                ans_end_idx = token_end_index + 1
            
            
            qap_score_search_offset = 5
            
            # ans_r.append(cls_prob)
            m_start = np.clip(ans_start_idx-qap_score_search_offset, 0, start_logits.shape[0])
            m_end = np.clip(ans_end_idx+qap_score_search_offset, 0, start_logits.shape[0])
            start_logits = start_logits[m_start: m_end].unsqueeze(0)
            end_logits = end_logits[m_start:m_end].unsqueeze(0)
            mm = torch.transpose(start_logits, 0, 1).mul(end_logits)
            if token_start_index > 510:
                print(token_start_index, token_end_index, ans_start_idx, ans_end_idx)

            # print(mm)
            geometric_ave = mm.max()
            r =  torch.sqrt(geometric_ave) 
            ans_probs.append(r)
    return max(ans_probs) if len(ans_probs) else None, min(cls_probs)


def get_all_question_candidate_answers(file_path,  device_no, tokenizer, all_encodings, dev_ds, qa_tokenizer, qa_answer_model, answer_verification_model,  include_no_answer=True, re_calc=False, add_null_candidate=False):
    with torch.no_grad():
        file_path = file_path #f"{dataset_subset}_all_question_candidate_answers_bert-large.pt"
        if os.path.exists(file_path) and not re_calc:
            all_question_candidate_answers = torch.load(file_path)
        else:
            all_question_candidate_answers = []
            # aggrement after validation on test set.
            for i, item in enumerate(tqdm.tqdm(dev_ds)):
                for para in item['paragraphs']:
                    src = para['context']                    
                    storyid = hashlib.md5(src.encode('utf-8')).hexdigest()
                    sents = sent_tokenize(src)
                    
                    encodings = all_encodings[storyid]
                    offsetmaps = [_[0] for _ in encodings]
                    encodings = [_[1].squeeze() for _ in encodings]

                    for j, question in enumerate(para['qas']):
                        answers = question['answers']
                        consensus = None
                        if len(answers) > 2:
                            answer_bounds = [(_['answer_start'], _['answer_start']+len(_['text'])) for _ in answers]
                            answer_counts = {f'{la[0]}-{la[1]}':answer_bounds.count(la) for la in answer_bounds}
                            answer_counts = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
                            # we only keep answers with disagreement for testing.
                            if len(answer_counts) > 1 and answer_counts[0][1] > answer_counts[1][1]:                                
                                consensus = {'s': int(answer_counts[0][0].split('-')[0]), 'e': int(answer_counts[0][0].split('-')[1])}
                        
                        if consensus is None:
                            continue
                        
                        target_ans_idx = -1
                        no_ans = False
                        ans_encodings = []
                        all_ans_scores = []
                        no_ans_idx = []                        
                        consensus_ans_text = None
                        all_ans_text = []
                        ans_idx = []

                        qat = qa_tokenizer(question['question'], src, 
                                        max_length=512, truncation="only_second", return_tensors='pt',                    
                                        stride=128,
                                        return_overflowing_tokens=True,
                                        padding="max_length",
                                        return_offsets_mapping=True)
                        sample_mapping = qat.pop("overflow_to_sample_mapping")
                        offset_mapping = qat.pop('offset_mapping')
                        qat = qat.to(qa_device)
                        qa_out = qa_answer_model(**qat)             
                        answer_confidences = []

                        scores = [0, 1]


                        consensus_ans_text = src[consensus['s']:consensus['e']].strip()
                        # aggregation after validation. remove duplicate answers.
                        encoded_answers = set()
                        for answer in answers:
                            ans = answer
                            assert src[answer['answer_start']:answer['answer_start']+len(answer['text'])] == answer['text']
                            ans['s'] = ans['answer_start']
                            ans['e'] = ans['answer_start'] + len(ans['text'])
                            ans_key = f"{ans['s']}-{ans['e']}"
                            if ans_key in encoded_answers:
                                continue
                            else:
                                encoded_answers.add(ans_key)

                            ans_prob, cls_prob = ans_confidence(qa_tokenizer, offset_mapping, qat, ans, qa_out)
                            ans_token_encodings = []

                            for sub_id, tok_map in enumerate(offsetmaps):                            
                                for tok_idx, token in enumerate(tok_map):
                                    # token in answer span                                
                                    if ans['s'] <= token[1] and ans['e'] >= token[0]:
                                        ans_token_encodings.append( encodings[sub_id][tok_idx,:])
                            try:
                                ans_token_encodings = torch.cat([_.unsqueeze(0) for _ in ans_token_encodings] , dim=0)
                            except:
                                print(len(ans_token_encodings), ans, len(src))
                                break
                            # print(ans_token_encodings.shape)
                            ans_pooled_encoding = torch.mean(ans_token_encodings, dim=0)
                            
                            if not no_ans and ans['s'] <= consensus['e'] and ans['e'] >= consensus['s']:
                                _x = len(src[consensus['s']:consensus['e']].split())
                                if len(src[ans['s']:ans['e']].split()) in range(_x-3, _x+3):
                                    ans_idx.append(len(ans_encodings))
                            ans_encodings.append(ans_pooled_encoding)
                            answer_sentence = src
                            if len(src[:ans['s']].split()) > 200:
                                start_char = 0
                                for i, sent in enumerate(sents):
                                    end_char = start_char + len(sent)
                                    if start_char <= ans['e'] and end_char > ans['s']:
                                        answer_sentence = sent
                                    start_char = end_char
                            ans_text = src[ans['s']:ans['e']].strip().lower()
                            all_ans_text.append(ans_text)
                            t = tokenizer('[SEP]'.join([question['question'].lower(), answer_sentence.lower()]), ans_text, max_length=512, truncation="only_first", return_tensors='pt').to(device_no)
                            out = answer_verification_model(**t)
                            out.logits = out.logits.cpu().detach()
                            ans_score = torch.nn.functional.softmax(out.logits, dim=-1).squeeze()   
                            all_ans_scores.append(ans_score)             
                            answer_confidences.append(ans_prob)     
                        cls_encoding = torch.mean(torch.stack([_[0,:] for _ in encodings]), dim=0)
                            
                        all_question_candidate_answers.append([no_ans, scores, ans_encodings, all_ans_scores, ans_idx, no_ans_idx, consensus_ans_text, all_ans_text, answer_confidences, cls_prob, cls_encoding])   

        torch.save(all_question_candidate_answers, file_path)    
    return all_question_candidate_answers


def main(args):
    dev_ds = args.dev_ds # "nq_dev.json"
    dev_ds = json.load(open(dev_ds))['data']
    dataset_subset = args.dataset_subset
    model_name = args.model_name 

    model_name_in_path = model_name.replace('/', '-')
    all_encodings = get_all_encodings(f"{dataset_subset}_{model_name_in_path}_encoding.pt", model_name, dev_ds)
    free_answer_encoding = get_all_candidate_answers_as_free_sequence(f"{dataset_subset}_all_answer_candidate_embeddings_{model_name_in_path}.pt", model_name, all_encodings, device_no, dev_ds, include_no_answer=False, re_calc=False, add_null_candidate=True)

    answer_verification_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model_path).to(device_no).eval()  
    tokenizer =  AutoTokenizer.from_pretrained(model_name, use_fast=True)

    
    qa_answer_model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model_path).eval().to(qa_device)
    qa_tokenizer = AutoTokenizer.from_pretrained(args.qa_model_path)    

    model_name_in_path = model_name.replace('/', '-')
    all_question_candidate_answers_with_null_candidate = get_all_question_candidate_answers(f"{dataset_subset}_all_question_candidate_answers_qa_no_noanswer_add_null_{model_name_in_path}.pt", device_no, tokenizer, all_encodings, , dev_ds, qa_tokenizer, qa_answer_model, answer_verification_model, include_no_answer=False, re_calc=False, add_null_candidate=True)

if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('--model_name', help='Encoder Name')
    params.add_argument('--dataset_subset', choices=['train', 'dev', 'test'], default='dev', help='subset name')
    params.add_argument('--dev_ds', type=str, default="nq_dev.json", help='path to the squad format data file.')
    params.add_argument('--nli_model_path', type=str, default="../caq-relevance/models/ckpt/nq_answer/ckpt-15000", help='path to the trained NLI model.')
    params.add_argument('--qa_model_path', type=str, default="../question_answering/nq/bert-base-uncased/checkpoint-10000", help='path to the trained QA model.')
    
    
    args = params.parse_args()

    main(args)
