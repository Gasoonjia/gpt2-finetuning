# DATA_FILE=/home/gasoon/research/offline-rl/exp/GPT2-finetune/cmu-book/dataset/booksummaries.txt
DATA_FILE=/home/gasoon/research/offline-rl/exp/GPT2-finetune/cmu-book/train.txt
EVAL_FILE=stas/openwebtext-10k
OUTPUT_DIR=/home/gasoon/research/offline-rl/exp/GPT2-finetune/cmu-book/train_output

python run_gpt2_recadam.py \
--output_dir=$OUTPUT_DIR \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--dataset_name $EVAL_FILE \
--do_eval \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=2 \
--learning_rate 5e-5 \
--num_train_epochs=10 \
--overwrite_output_dir \
--evaluation_strategy epoch  \
--validation_file=$DATA_FILE \
--do_train \
--train_file=$DATA_FILE \
--optimizer_type RecAdam \
--cache_dir ~/code/.transformer_cache/ \
--recadam_anneal_k 5e-4 \
--recadam_anneal_t0 0 \
# --logging_strategy epoch \
# --evaluate_during_training \
# --line_by_line \