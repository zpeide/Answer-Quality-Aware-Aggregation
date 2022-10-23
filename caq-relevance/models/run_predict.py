import numpy as np
from random import shuffle
import torch
from torch.utils.data import (DataLoader, SequentialSampler)
import json
import os
import tqdm
import argparse
from transformers import BertForSequenceClassification, BertTokenizerFast, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils import SeqClsDatasetForBert, batch_list_to_batch_tensors


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_file", default=None, type=str, required=True,
                        help="data (json format) for testing. Keys: source and target")

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")

    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--device", default=0, type=int,
                        help="gpu device.")

    parser.add_argument("--use_ans", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    args = parser.parse_args()
    return args


def predict(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path) # BertForSequenceClassification.from_pretrained(args.model_name_or_path)
    model = model.eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir="../../cache/") #, use_fast=True) # BertTokenizerFast.from_pretrained("bert-base-uncased")

    dataset = SeqClsDatasetForBert(args.eval_file, 0, shuffle=False, use_ans=args.use_ans)

    dataloader = DataLoader(
        dataset, sampler=SequentialSampler(dataset),
        batch_size=args.batch_size)


    results = []
    ground_truth = []
    cnt = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm.tqdm(dataloader)):
            encoding, labels = batch_list_to_batch_tensors(tokenizer, batch)
            encoding.to(args.device)
            out = model(**encoding)
            # print(out)
            results.append( torch.nn.functional.softmax(out.logits).argmax(-1).detach().cpu())
            ground_truth.append(labels)


    ground_truth = torch.cat(ground_truth, dim=0)
    preds = torch.cat(results, dim=0)

    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, preds, average='binary')
    acc = accuracy_score(ground_truth, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == "__main__":
    args = get_args()

    results = predict(args)

    print(results)
