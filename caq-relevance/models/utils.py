import json
import torch

import logging
import os
import json
import random

logger = logging.getLogger(__name__)


class SeqClsDatasetForBert(torch.utils.data.Dataset):
    def __init__(self, file_path, offset=0, shuffle=True, use_ans=False):
        self.items = json.load(open(file_path)) #[json.loads(_) for _ in open(file_path)]
        self.offset = offset
        self.use_ans = use_ans
        if offset > 0:
            logger.info("  ****  Set offset %d in SeqClsDatasetForBert ****  ", offset)
        if shuffle:
            random.shuffle(self.items)
    def __len__(self):
        return len(self.items) - self.offset

    def __getitem__(self, idx):
        idx = self.offset + idx
        # q1, , label
        if self.use_ans:
            return '[SEP]'.join([self.items[idx]['q'].lower(), self.items[idx]['src'].lower()]), self.items[idx]['ans'].lower(), self.items[idx]['label']
        else:
            return self.items[idx]['src'].lower(), self.items[idx]['q'].lower(), self.items[idx]['label']
        


def batch_list_to_batch_tensors(tokenizer, batch):
    # batch[0]: q1, batch[1]: q2, batch[2]: label
    encoding = tokenizer(batch[0], batch[1], return_tensors='pt', padding=True, truncation="only_first")
    labels = torch.tensor(batch[2]).long()

    return encoding, labels
