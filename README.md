
code for [Answer Quality Aware Aggregation for Extractive QA Crowdsourcing](https://preview.aclanthology.org/emnlp-22-ingestion/2022.findings-emnlp.457.pdf)

## Citation
```
@inproceedings{zhu-hauff-2022-unsupervised,
    title = "Unsupervised Domain Adaptation for Question Generation with {D}omain{D}ata Selection and Self-training",
    author = "Zhu, Peide  and
      Hauff, Claudia",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.183",
    doi = "10.18653/v1/2022.findings-naacl.183",
    pages = "2388--2401",
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



