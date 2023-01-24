
code for [Answer Quality Aware Aggregation for Extractive QA Crowdsourcing](https://preview.aclanthology.org/emnlp-22-ingestion/2022.findings-emnlp.457.pdf)

## Citation
```
@inproceedings{zhu-hauff-2022-AnswerQuality,
    title = " Answer Quality Aware Aggregation for Extractive QA Crowdsourcing",
    author = "Zhu, Peide and Wang, Zhen and Hauff, Claudia and Yang, Jie  and Anand, Avishek",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = December,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://preview.aclanthology.org/emnlp-22-ingestion/2022.findings-emnlp.457/",
    doi = "10.18653/v1/2022.findings-naacl.183",
    pages = "",
    abbr={EMNLP},
    selected=true,
    pdf={https://preview.aclanthology.org/emnlp-22-ingestion/2022.findings-emnlp.457.pdf},
    code={https://github.com/zpeide/Answer-Quality-Aware-Aggregation},
    abstract = "Quality control is essential for creating extractive question answering (EQA) datasets via crowdsourcing. Aggregation across answers, i.e. word spans within passages annotated, by different crowd workers is one major focus for ensuring its quality. However, crowd workers cannot reach a consensus on a considerable portion of questions. We introduce a simple yet effective answer aggregation method that takes into account the relations among the answer, question, and context passage. We evaluate answer quality from both the view of question answering model to determine how confident the QA model is about each answer and the view of the answer verification model to determine whether the answer is correct. Then we compute aggregation scores with each answer’s quality and its contextual embedding produced by pre-trained language models. The experiments on a large real crowdsourced EQA dataset show that our framework outperforms baselines by around 16% on precision and effectively conduct answer aggregation for extractive QA task.",
}
```

## Create NEWSQA dataset
- Please refer [Maluuba NewsQA Tool](https://github.com/Maluuba/newsqa.git) for creating NEWSQA dataset
- make all splits
    ```
    create_newsqa_splits.py 
   ```

## Question Answering   (part of the code is adopted from [HuggingFace])
   change to the question_answering folder.

   - Train QA Model.
    
   ```bash
    ./run_newsqa_train.sh
   ```
   -  Test on primary consensus test 
   
   ```bash
    ./run_newsqa_eval.sh  newsqa_qa/bert-base/checkpoint-8500/
   ```

   - Test on test_consensus_after_validate.json
    
   ```bash
    ./run_newsqa_eval_val.sh  newsqa_qa/bert-base/checkpoint-8500/
   ```

## Train the answer verification model.
 change to the caq-relevance folder.
 

### Question Answering using the aggregated data.
  - Random
  
   ``` 
    ./run_newsqa_random_train.sh
   ```

  - Ground Truth
  
   ```
    ./run_newsqa_ground_truth_train.sh
   ```

  - ACAF-SMV
   
   ```
    ./run_newsqa_acaf_smv_train.sh
   ```

  - ACAF-SMS
   
  ```
    ./run_newsqa_acaf_sms_train.sh
  ```

   - ACAF-SMV Voted
    
   ```
    ./run_newsqa_acaf_smv_vote_train.sh
   ```

   - ACAF-SMS Voted
   
   ```
    ./run_newsqa_acaf_sms_vote_train.sh
   ```


## Experiment on Natural Questions
- change to the NQ directory
- Convert Natural Questions to SQuAD format.
  ```
  nq_to_squad.py    --nq_file       path to natural question dataset (.jsonl file)
                    --output_file   path to the generated squad format file.
                    --do_train      whether it is the training set.
  ``` 
- Pre-compute all embedding files for aggregation.

  ```
  create_embedding_file.py
  ``` 
  
- Answer aggregation:

  ```
    answer_aggregation.py
  ```



