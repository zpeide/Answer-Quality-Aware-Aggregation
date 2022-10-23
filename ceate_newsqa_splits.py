import json
import numpy as np
import os, sys
import argparse
import spacy
import random
import tqdm


# qa items
def get_consus_stage_1_questions(ds):
    consensus_qa_items_stage_1 = {'id':[], 'context':[], 'question':[], 'answers':[]}
    consensus_after_validate = {'id':[], 'context':[], 'question':[], 'answers':[]} # _  for _ in validate_questions if len(_['validatedAnswers']) <= 2] # 'consensus' in _]
    no_consensus_after_validate = {'id':[], 'context':[], 'question':[], 'answers':[]} #_  for _ in validate_questions if len(_['validatedAnswers']) > 2]     
    
    for item in ds:
        text = item['text']
        type = item['type']
        questions = item['questions']
        storyId = item['storyId']
        for qidx, question in enumerate(questions):
            qid = '#'.join([storyId, str(qidx)])
            if 'validatedAnswers' not in question:
                # stage 1
                cons = question['consensus']
                if 's' in cons:
                    # consensused on answer
                    answers = {
                        'answer_start': [cons['s']],
                        'text': [ text[cons['s']: cons['e'] ] ],
                    }
                else:
                    answers = {
                        'answer_start': [],
                        'text': [],
                    }
                consensus_qa_items_stage_1['id'].append(qid)
                consensus_qa_items_stage_1['context'].append(text)
                consensus_qa_items_stage_1['question'].append( question['q'])
                consensus_qa_items_stage_1['answers'].append(answers)
            else:
                # stage 1
                cons = question['consensus']
                if 's' in cons:
                    # consensused on answer
                    answers = {
                        'answer_start': [cons['s']],
                        'text': [ text[cons['s']: cons['e'] ] ],
                    }
                else:
                    answers = {
                        'answer_start': [],
                        'text': [],
                    }
                if len(question['validatedAnswers']) <= 2:                    
                    consensus_after_validate['id'].append(qid)
                    consensus_after_validate['context'].append(text)
                    consensus_after_validate['question'].append( question['q'])
                    consensus_after_validate['answers'].append(answers)
                else:
                    no_consensus_after_validate['id'].append(qid)
                    no_consensus_after_validate['context'].append(text)
                    no_consensus_after_validate['question'].append( question['q'])
                    no_consensus_after_validate['answers'].append(answers)                    
    return consensus_qa_items_stage_1, consensus_after_validate, no_consensus_after_validate


def parsed_process(newsqa, directory):
    nlp = spacy.load("en_core_web_sm")
    
    dataset = {'train':[], 'dev':[], 'test':[]}

    for item in newsqa:
        dataset[item['type']].append(item)

    for typek, items in dataset.items():
        src_question_items = []
        src_answer_items = []

        story_texts = {}
    
        for _ij, item in enumerate(tqdm(items)):
            story_id = item['storyId']
            text = item['text']

            doc = nlp(text)
            tokens = [_ for _ in doc]
            sents = [s for s in doc.sents]
            
            questions = item['questions']

            for question in questions:
                q = question['q']
                consensus = question['consensus']
                # only consensus in stage 1
                if 'validatedAnswers' not in question:    
                    if 'noAnswer' in consensus or 'badQuestion' in consensus:
                        src_question_items.append({
                            'storyid':story_id,
                            'src':text,
                            'q': q,
                            'label': 0
                        })
                        ans_dict = {}
                        for answer in question['answers']:
                            for ans in answer['sourcerAnswers']:
                                if 's' in ans:
                                    ans_text = text[ans['s']:ans['e']].strip()
                                    if ans_text not in ans_dict:
                                        ans_dict[ans_text] = 1                                        
                                        src_answer_items.append({
                                            'storyid':story_id,
                                            'src': text,
                                            'q': q,
                                            'ans': ans_text,
                                            'start': ans['s'],
                                            'label': 0,
                                        })
                    else:
                        # entailed question
                        src_question_items.append({
                            'storyid':story_id,
                            'src':text,
                            'q': q,
                            'label': 1
                        })
                        # print(consensus)
                        ans_text = text[consensus['s']:consensus['e']].strip()
                        ans_dict = {ans_text: consensus['s']}
                        answer_sentence = text
                        if len(text[:consensus['s']].split()) > 200:
                            # get answer sentence
                            for i, sent in enumerate(sents):
                                if sent.start_char <= consensus['e'] and sent.end_char > consensus['s']:
                                    answer_sentence = ' '.join([_.text for _ in sent]).strip()

                        src_answer_items.append({
                            'storyid':story_id,                        
                            'src': answer_sentence,
                            'q': q,
                            'ans': ans_text,
                            'label': 1,

                        })                    
                        for answer in question['answers']:
                            for ans in answer['sourcerAnswers']:
                                if 's' in ans:
                                    answer_sentence = text
                                    if len(text[:ans['s']].split()) > 200:
                                        for i, sent in enumerate(sents):
                                            if sent.start_char <= ans['e'] and sent.end_char > ans['s']:
                                                answer_sentence = ' '.join([_.text for _ in sent]).strip()

                                    ans_text = text[ans['s']:ans['e']].strip()
                                    if ans_text in ans_dict:
                                        if ans_dict[ans_text] == ans['s']:
                                            continue
                                        else:
                                            src_answer_items.append({
                                                'storyid':story_id,
                                                'src': answer_sentence,
                                                'q': q,
                                                'ans': ans_text,
                                                'label': 0,
                                            })                                    
                                    else:
                                        src_answer_items.append({
                                            'storyid':story_id,
                                            'src': answer_sentence,
                                            'q': q,
                                            'ans': ans_text,
                                            'label': 0,
                                        })            
                                        
                                    ans_dict[ans_text] = ans['s']
            story_texts[story_id] = doc
        json.dump(src_question_items, open(f"{directory}/{typek}/src_question_items.json", "w"))
        json.dump(src_answer_items, open(f"{directory}/{typek}/src_question_answer_items.json", "w"))
        np.savez(f"{directory}/{typek}/story_texts.npz", story_texts, allow_pickle=True)


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--infile",
        type=str,
        default="../data/newsqa/combined-newsqa-data-v1.json",
        help="input raw file."
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=21,
    )


    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    directory = os.path.basename(args.infile)
    ds = json.load(open(args.infile))

    ds = ds['data']
    train = [_ for _ in ds if _['type'] == 'train']
    dev = [_ for _ in ds if _['type'] == 'dev']
    test = [_ for _ in ds if _['type'] == 'test']

    train_consensus, train_consensus_after_validate, train_no_consensus = get_consus_stage_1_questions(train)
    dev_consensus, dev_consensus_after_validate, dev_no_consensus = get_consus_stage_1_questions(dev)
    test_consensus, test_consensus_after_validate, test_no_consensus = get_consus_stage_1_questions(test)

    directory = "data/newsqa/"
    json.dump( {'data': train_consensus}, open(os.path.join(directory, "train_consensus.json"), 'w') )
    json.dump( {'data': train_consensus_after_validate}, open(os.path.join(directory, "train_consensus_after_validate.json"), 'w') )
    json.dump( {'data': train_no_consensus}, open(os.path.join(directory, "train_no_consensus.json"), 'w') )

    json.dump( {'data': dev_consensus}, open(os.path.join(directory, "dev_consensus.json"), 'w') )
    json.dump( {'data': dev_consensus_after_validate}, open(os.path.join(directory, "dev_consensus_after_validate.json"), 'w') )
    json.dump( {'data': dev_no_consensus}, open(os.path.join(directory, "dev_no_consensus.json"), 'w') )

    json.dump( {'data': test_consensus}, open(os.path.join(directory, "test_consensus.json"), 'w') )
    json.dump( {'data': test_consensus_after_validate}, open(os.path.join(directory, "test_consensus_after_validate.json"), 'w') )
    json.dump( {'data': test_no_consensus}, open(os.path.join(directory, "test_no_consensus.json"), 'w') )





