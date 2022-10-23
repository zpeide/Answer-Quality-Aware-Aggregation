import torch
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
import spacy 
import json
import numpy as np
from matplotlib import pyplot as plt
import os, sys


def main(args):
    nlp = spacy.load("en_core_web_sm")
    ds = json.load(open(args.infile))['data']
    
    max_src_token = 352

    src_question_items = []
    src_answer_items = []

    
    
    for _ij, item in enumerate(tqdm(ds)):        
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            doc = nlp(context)
            tokens = [_ for _ in doc]
            sents = [s for s in doc.sents]
            ent_types = [_.pos_ for _ in tokens]
            pos_types = [_.ent_type_ for _ in tokens]
            ent_idx_map = {}
            pos_idx_map = {}
            for i, token in enumerate(tokens):
                if token.ent_type_ != '':                    
                    ent_idx_map[token.ent_type_] = ent_idx_map.get(token.ent_type_, []) + [i]

            for qa in paragraph['qas']:
                question = qa['question']
                answers = qa['answers']
                qid = qa['id']
                for ans in answers:
                    st = ans['answer_start']
                    ans_tok = []
                    ans_idx = []
                    ans_ents = []
                    for i, tok in enumerate(tokens):
                        if tok.idx >= st and tok.idx < st + len(ans['text']):
                            ans_tok.append(tok)
                            ans_idx.append(i)
                            
                            if tok.ent_type_ != '' and len(ent_idx_map[tok.ent_type_]) > 1:
                                ans_ents.append(tok)
                                
                    if len(ans_tok) == 0:
                        print(context[ans['answer_start']-20:])
                        print(_ij, qa)
                        continue
                    neg_answer_toks = []
                    # make negative answer
                    # 1. if has an entity
                    negative_ans = []
                    ans_tok_text = [_.text for _ in ans_tok]
                    for ae in ans_ents:
                        if len(ent_idx_map[ae.ent_type_]) > 1:
                            for e in ent_idx_map[ae.ent_type_]:
                                if e < ans_idx[0] and e > ans_idx[-1]:
                                    if tokens[e].text not in ans_tok_text:
                                        negative_ans.extend(tokens[e:e+len(ans_tok)])
                                        print(f"NE---------;{ans['text']}---{' '.join([_.text for _ in tokens[e:e+len(ans_tok)]])}")
                    if len(negative_ans) == 0:
                        ans_pos = set([_.pos_ for _ in ans_tok])
                        spans = {}
                        for i in range(0, ans_idx[0] - len(ans_tok)):
                            x = set([_.pos_ for _ in tokens[i:i+len(ans_tok)]])
                            spans[i] = len(x.intersection(ans_pos))
                        
                        for i in range(ans_idx[-1], len(tokens)):
                            x = set([_.pos_ for _ in tokens[i:i+len(ans_tok)]])
                            spans[i] = len(x.intersection(ans_pos))
                        spans = sorted(spans.items(), key=lambda x: x[1], reverse=True)
                        # if ans['text'].strip() == 'Dublin':
                        #     print(spans)
                        #     print([_.pos_ for _ in ans_tok])
                        #     print([_.ent_type_ for _ in ans_tok])
                        for span in spans:
                            span_tok = tokens[span[0]:span[0]+len(ans_tok)]
                            span_text = ' '.join([_.text for _ in span_tok])
                            if span_text != ans['text']:
                                negative_ans.append( span_tok)
                                break
                    src_answer_items.append({
                        'id':qid,
                        'src': context,
                        'q': question,
                        'ans': ans['text'],
                        'start': ans['answer_start'],
                        'label': 1,
                    })                                          
                    
                    
                    src_answer_items.append({
                        'id':qid,
                        'src': context,
                        'q': question,
                        'ans': ' '.join([_.text for _ in negative_ans[0]]),
                        'start': negative_ans[0][0].idx,
                        'label': 0,
                    })                                          
                    # print(ans['text'], len(ans_tok))
                    # print(' '.join([_.text for _ in negative_ans[0]]))
    
        
    json.dump(src_answer_items, open(args.outfile, "w"))
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--infile",
        type=str,
        default="../NQ/nq_train.json",
        help="input raw file."
    )

    parser.add_argument(
        "--outfile",
        type=str,
        default="../NQ/src_question_answer_items.json",
        help="output raw file."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=21,
    )


    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
