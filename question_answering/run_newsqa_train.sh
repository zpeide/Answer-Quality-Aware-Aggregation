python run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name newsqa \
    --train_file  ../data/newsqa/train_consensus.json \
    --validation_file ../data/newsqa/dev_consensus.json  \
    --do_train \
    --per_device_train_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --overwrite_output_dir \
    --output_dir ./newsqa_qa/bert-base/ \
    --use_itemrized_file \
    --version_2_with_negative    
    