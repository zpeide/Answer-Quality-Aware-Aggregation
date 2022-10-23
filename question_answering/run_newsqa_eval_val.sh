MODEL_DIR=$1
python run_qa.py \
    --model_name_or_path ${MODEL_DIR}\
    --dataset_name newsqa \
    --train_file  ../data/newsqa/train_consensus.json \
    --validation_file ../data/newsqa/test_consensus_after_validate.json \
    --do_eval \
    --per_device_train_batch_size 8 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ${MODEL_DIR} \
    --use_itemrized_file \
    --version_2_with_negative

# test_consensus_answered_only.json \