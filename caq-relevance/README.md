# Relevance of Content-Answer-Reference and the generations

## Create dataset for training.

### Trainset
```bash
 python create_dataset.py 
```

## Train on Question Answerability 
nohup ./run_train_ans.sh &

## Train on Answer Verification
nohup ./run_train.sh &


##  Prediction

### Prediction on question answerability
```
python run_predict.py --eval_file ../../data/newsqa/test/src_question_items.json  --model_name_or_path ckpt/question/ckpt-6000/ --batch_size 32 --tokenizer_name bert-base-uncased
```
{'accuracy': 0.8376068376068376, 'f1': 0.8850406843313164, 'precision': 0.8528347406513872, 'recall': 0.9197745013009541}

### Prediction on answer verification
```
python run_predict.py --eval_file ../../data/newsqa/test/src_question_answer_items.json --model_name_or_path ckpt/answer/ckpt-6000/ --batch_size 32 --use_ans
```
{'accuracy': 0.7037037037037037, 'f1': 0.7676842889054355, 'precision': 0.6725375081539465, 'recall': 0.8941890719861232}
{'accuracy': 0.8065112311629229, 'f1': 0.8758777929776561, 'precision': 0.8515694272034049, 'recall': 0.9016147202403304}

python run_predict.py --eval_file ../../data/newsqa/test/src_question_answer_items.json --model_name_or_path ckpt/answer_bert_large/ckpt-19500/ --batch_size 32 --use_ans --tokenizer_name bert-base-uncased
{'accuracy': 0.6923076923076923, 'f1': 0.7688191223688905, 'precision': 0.6530303030303031, 'recall': 0.9345186470078057}



# NaturalQuestions Dataset
```
python create_nq_dataset.py --infile ../NQ/train.json --outfile ../NQ/src_question_answer_items_train.json

python  run_predict.py --model_name_or_path ckpt/nq_answer/ckpt-10500/ --use_ans --eval_file ../../NQ/src_question_answer_items.dev.json --tokenizer_name bert-base-uncased --batch_size 32


{'accuracy': 0.944623742671185, 'f1': 0.9461854107315544, 'precision': 0.9202338586680223, 'recall': 0.973643160669141}
```
# SQuAD dataset
```
python create_nq_dataset.py --infile ../data/train-v1.1.json --outfile squad/src_question_answer_items_train.json > squad.log

python  run_predict.py --model_name_or_path ckpt/squad_answer/ckpt-10500/ --use_ans --eval_file ../../squad/src_question_answer_items.dev.json --tokenizer_name bert-base-uncased --batch_size 32
{'accuracy': 0.9552690999514105, 'f1': 0.9562127528608602, 'precision': 0.9364313897413415, 'recall': 0.9768478820099469}
```
