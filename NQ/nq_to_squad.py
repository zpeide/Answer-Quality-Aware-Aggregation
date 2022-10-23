import json
import argparse
import gzip
import glob
import numpy as np
import tqdm


DEBUG = False
debug_print = lambda *x: print(x) if DEBUG else None



def nq_train_item(nq_example):
    tokens = nq_example['document_text'].split()
    annotation = nq_example['annotations'][0]
    long_answer = annotation['long_answer']
        
    if tokens[long_answer['start_token']] != '<P>' or tokens[long_answer['end_token'] - 1] != '</P>':
        return None
    else:                
        long_tokens = tokens[long_answer['start_token']+1: long_answer['end_token']-1]
        long_text = ' '.join(long_tokens)
        short_answers = annotation['short_answers']    
        if len(short_answers) == 0:
            return None
        processed_short_answers = []
        for ans in short_answers:
            start_token = ans['start_token'] - (long_answer['start_token']+1)
            answer_start = len(' '.join(long_tokens[:start_token])) + (start_token != 0)
            answer_text = ' '.join(tokens[ans['start_token']:ans['end_token']])
            debug_print(long_text)
            debug_print(answer_text)
            debug_print(long_text[answer_start:answer_start+len(answer_text)])
            assert long_text[answer_start:answer_start+len(answer_text)] == answer_text
            processed_short_answers.append({'text':answer_text, 'answer_start': answer_start})
        return ' '.join(long_tokens), processed_short_answers
        

def nq_dev_item(nq_example):
    tokens = [_['token'] for _ in nq_example['document_tokens']]
    if len(nq_example['annotations']) == 0:
        return None

        
    long_answers = [a['long_answer'] for a in nq_example['annotations'] if a['long_answer']['start_byte'] >= 0]
    long_answer_bounds = [(la['start_byte'], la['end_byte']) for la in long_answers]
    long_answer_counts = [long_answer_bounds.count(la) for la in long_answer_bounds]
    if len(long_answer_counts) == 0:
        return None
    long_answer = long_answers[np.argmax(long_answer_counts)]
    
    if tokens[long_answer['start_token']] != '<P>' or tokens[long_answer['end_token'] - 1] != '</P>':
        return None
    short_answers = []
    for annotation in nq_example['annotations']:
        answer = annotation['long_answer']
        if answer['candidate_index'] != long_answer['candidate_index']:
            continue
        short_answers.extend(annotation['short_answers'])
    long_tokens = tokens[long_answer['start_token']+1: long_answer['end_token']-1]
    long_text = ' '.join(long_tokens)
    processed_short_answers = []
    
    for ans in short_answers:
        start_token = ans['start_token'] - (long_answer['start_token']+1)
        answer_start = len(' '.join(long_tokens[:start_token])) + (start_token != 0)
        answer_text = ' '.join(tokens[ans['start_token']:ans['end_token']])
        debug_print(long_text)
        debug_print(answer_text)
        debug_print(long_text[answer_start:answer_start+len(answer_text)])
        assert long_text[answer_start:answer_start+len(answer_text)] == answer_text
        processed_short_answers.append({'text':answer_text, 'answer_start': answer_start})

    return long_text, processed_short_answers        


def nq_to_squad_format(filename, output_file, do_train=True):
    data = []
    # for filename in glob.glob(nq_dir + '/*.jsonl'):
        # with gzip.open(filename, 'r') as f:
    with open(filename, 'r') as f:
        for line in f:
            nq_example = json.loads(line)
            if do_train:
                item = nq_train_item(nq_example)
            else:
                item = nq_dev_item(nq_example)
            if item is None:
                continue
            question_text = nq_example['question_text']
            context = item[0]
            para = {'context': context, 'qas': [{'question': question_text, 'answers': []}]}
            data.append({'paragraphs': [para]})
            qa = para['qas'][0]
            qa['id'] = str(nq_example['example_id'])
            qa['is_impossible'] = True
            short_answers = item[1]

            if len(short_answers) > 0:
                qa['answers'].extend(short_answers)
                qa['is_impossible'] = False

    nq_as_squad = {'data': data, 'version': '2.0'}

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(json.dumps(nq_as_squad, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('--nq_file', help='path to NQ .jsonl file')
    params.add_argument('--output_file', help='Output file in SQuAD format', default="nq_dev.json")
    params.add_argument('--do_train', action='store_true', default=False, help='process the train set.')
    args = params.parse_args()

    nq_to_squad_format(args.nq_file, args.output_file, args.do_train)
