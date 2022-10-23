# path of training data
# export MASTER_PORT=3665
TRAIN_FILE="../../data/newsqa/train/src_question_answer_items_2.json"
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=./ckpt/answer_2/
# folder used to cache package dependencies
CACHE_DIR=./cache
# python -m torch.distributed.launch --nproc_per_node=4 run_train.py --train_file ${TRAIN_FILE} \
python run_train.py --train_file ${TRAIN_FILE} \
            --output_dir ${OUTPUT_DIR} \
            --model_type bert-base-uncased \
            --model_name_or_path bert-base-uncased  \
            --do_lower_case \
            --max_source_seq_length 464 \
            --max_target_seq_length 48   \
            --per_gpu_train_batch_size 8 \
            --gradient_accumulation_steps 1   \
            --learning_rate 1e-5 \
            --num_warmup_steps 5000 \
            --num_training_steps 320000 \
            --cache_dir ${CACHE_DIR} \
            --use_ans
