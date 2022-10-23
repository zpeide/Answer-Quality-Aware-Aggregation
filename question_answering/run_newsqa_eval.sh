MODEL_DIR=$1
python run_qa.py \
    --model_name_or_path ${MODEL_DIR}\
    --dataset_name newsqa \
    --train_file  ../data/newsqa/ground_truth_train.json \
    --validation_file ../data/newsqa/ground_truth_dev.json \
    --do_eval \
    --per_device_train_batch_size 4 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir ${MODEL_DIR} \
    --version_2_with_negative

