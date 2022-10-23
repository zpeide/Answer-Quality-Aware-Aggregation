import argparse

from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizerFast, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import torch

from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler

from utils import SeqClsDatasetForBert
from utils import batch_list_to_batch_tensors

import logging
import random
import numpy  as np
import os
import tqdm
# import wandb
import os


logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")

    parser.add_argument("--use_ans", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    ## Other parameters
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_source_seq_length", default=464, type=int,
                        help="The maximum total source sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_target_seq_length", default=48, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--cached_train_features_file", default=None, type=str,
                        help="Cached training features file")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_training_steps", default=-1, type=int,
                        help="set total number of training steps to perform")
    parser.add_argument("--num_training_epochs", default=10, type=int,
                        help="set total number of training epochs to perform (--num_training_steps has higher priority)")
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--random_prob", default=0.1, type=float,
                        help="prob to random replace a masked token")
    parser.add_argument("--keep_prob", default=0.1, type=float,
                        help="prob to keep no change for a masked token")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    args = parser.parse_args()
    return args



def train(args):

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device        



    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    # wandb.init(project='question-answerability', entity='pitter')
    recover_step = None
    if args.model_name_or_path is not None:
        c = os.path.dirname(args.model_name_or_path).split("ckpt-")
        if len(c) > 1:
            recover_step = int(c[1])
            print("train from recover step:{}".format(recover_step))
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path if args.model_name_or_path is not None else 'bert-base-uncased' ) # AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path if args.model_name_or_path is not None else 'bert-base-uncased' ) # 
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=True) # BertTokenizerFast.from_pretrained("bert-base-uncased")

    if args.local_rank == 0:
        model = model.to(0)
        torch.distributed.barrier()

    model.to(args.device)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if args.n_gpu == 0 or args.no_cuda:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    else:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.n_gpu * args.gradient_accumulation_steps

    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_training_steps, last_epoch=-1)

    
    train_dataset = SeqClsDatasetForBert(args.train_file, 0, True, use_ans=args.use_ans)
    # The training features are shuffled
    train_sampler = SequentialSampler(train_dataset) \
        if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=per_node_train_batch_size // args.gradient_accumulation_steps)

   
    model.train()
    model.zero_grad()

    global_step = 0
    logging_loss = 0

    if recover_step is not None:
        global_step = recover_step

    total_iterations = args.num_training_steps // (len(train_dataset) // per_node_train_batch_size)
    print("total iterations: ", total_iterations)
    for iteration in tqdm.tqdm(range(total_iterations), desc="iteration"):
        train_iterator = tqdm.tqdm(
                   train_dataloader, initial=0,
                   desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(train_iterator):
            #print(batch)
            encoding, labels = batch_list_to_batch_tensors(tokenizer, batch)
            outputs = model(**encoding, labels=labels) #.float())
            loss = outputs.loss
            if args.n_gpu > 1:
                loss = loss.mean()
            train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            logging_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logger.info("")
                logger.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
                # wandb.log({'loss': logging_loss, 'learning_rate': scheduler.get_lr()[0]})
                logging_loss = 0.0
            if args.local_rank in [-1, 0] and args.save_steps > 0 and \
                    (global_step % args.save_steps == 0 or global_step == args.num_training_steps):

                save_path = os.path.join(args.output_dir, "ckpt-%d" % global_step)
                os.makedirs(save_path, exist_ok=True)
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(save_path)

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logger.info("")
                logger.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
                logging_loss = 0.0

            if args.local_rank in [-1, 0] and args.save_steps > 0 and \
                    (global_step % args.save_steps == 0 or global_step == args.num_training_steps):

                save_path = os.path.join(args.output_dir, "ckpt-%d" % global_step)
                os.makedirs(save_path, exist_ok=True)
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(save_path)
                
                logger.info("Saving model checkpoint %d into %s", global_step, save_path)

        

if __name__ == '__main__':
    args = get_args()


    train(args)    

        

