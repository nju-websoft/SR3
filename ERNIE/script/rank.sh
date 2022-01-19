#!/usr/bin/env bash
set -eux

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/home/home1/ychen/ERNIE/pretrainmodel
export TASK_DATA_PATH=../../data
export LD_LIBRARY_PATH=/home/home1/ychen/.conda/envs/chenyue/lib
python -u ../run_classifier.py \
                   --use_cuda True \
                   --verbose true \
                   --do_train True \
                   --do_val True \
                   --do_test True \
                   --batch_size 32 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/train.tsv \
           		   --dev_set ${TASK_DATA_PATH}/val.tsv \
				   --test_set ${TASK_DATA_PATH}/test.tsv \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --checkpoints ./checkpoints \
				   --learning_rate 2e-5 \
                   --save_steps 1000 \
                   --weight_decay  0.0 \
                   --warmup_proportion 0.0 \
                   --validation_steps 1000 \
                   --epoch 1 \
                   --max_seq_len 128 \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --learning_rate 2e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1

python -u ../run_classifier.py \
                   --use_cuda True \
                   --verbose true \
                   --do_train False \
                   --do_val False \
                   --do_test True \
                   --batch_size 32 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/train.tsv \
           		   --dev_set ${TASK_DATA_PATH}/val.tsv \
				   --test_set ${TASK_DATA_PATH}/test.tsv \
				   --test_save ${TASK_DATA_PATH}/test_result.0.0 \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --init_checkpoint ./checkpoints/step_14175 \
                   --checkpoints ./checkpoints \
				   --learning_rate 2e-5 \
                   --save_steps 1000 \
                   --weight_decay  0.0 \
                   --warmup_proportion 0.0 \
                   --validation_steps 1000 \
                   --epoch 1 \
                   --max_seq_len 128 \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --learning_rate 2e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1

python -u ../run_classifier.py \
                   --use_cuda True \
                   --verbose true \
                   --do_train False \
                   --do_val False \
                   --do_test True \
                   --batch_size 32 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/train.tsv \
           		   --dev_set ${TASK_DATA_PATH}/val.tsv \
				   --test_set ${TASK_DATA_PATH}/val.tsv \
				   --test_save ${TASK_DATA_PATH}/val_result.0.0 \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --init_checkpoint ./checkpoints/step_14175 \
                   --checkpoints ./checkpoints \
				   --learning_rate 2e-5 \
                   --save_steps 1000 \
                   --weight_decay  0.0 \
                   --warmup_proportion 0.0 \
                   --validation_steps 1000 \
                   --epoch 1 \
                   --max_seq_len 128 \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --learning_rate 2e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1

python -u ../run_classifier.py \
                   --use_cuda True \
                   --verbose true \
                   --do_train False \
                   --do_val False \
                   --do_test True \
                   --batch_size 32 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/train.tsv \
           		   --dev_set ${TASK_DATA_PATH}/val.tsv \
				   --test_set ${TASK_DATA_PATH}/train.tsv \
				   --test_save ${TASK_DATA_PATH}/train_result.0.0 \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --init_checkpoint ./checkpoints/step_14175 \
                   --checkpoints ./checkpoints \
				   --learning_rate 2e-5 \
                   --save_steps 1000 \
                   --weight_decay  0.0 \
                   --warmup_proportion 0.0 \
                   --validation_steps 1000 \
                   --epoch 1 \
                   --max_seq_len 128 \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --learning_rate 2e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1