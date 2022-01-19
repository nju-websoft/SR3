#!/usr/bin/env bash
#################for ckgg
#python train_graph.py \
#    -split_qm True \
#    -copy_decoder True \
#    -copy_word True \
#    -encode_q True \
#    -mode train \
#    -graph_transformer True \
#    -use_cls True \
#    -avg_sent True \
#    -copy_sent True \
#    -gpu 4 \
#    -valid_step 500 \
#    -do_eval True \
#    -train_steps 20000 \
#    -save_checkpoint_steps 10000 \
#    -temp_dir /home/ychen/PreSumm/temp \
#    -train_batch_size 1500 \
#    -test_batch_size 1500 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path ckgg/bert_data/  \
#    -model_path ckgg/models_copysent_graph/ \
#    -result_path ckgg/results_copysent_graph/ \
#    -log_file ckgg/logs/log_copysent_graph \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\la
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

#python train_graph.py \
#    -split_qm True \
#    -copy_decoder True \
#    -copy_word True \
#    -encode_q True \
#    -mode train \
#    -graph_transformer True \
#    -use_cls True \
#    -avg_sent True \
#    -copy_sent True \
#    -gpu 4 \
#    -valid_step 500 \
#    -do_eval True \
#    -train_steps 20000 \
#    -save_checkpoint_steps 10000 \
#    -temp_dir /home/ychen/PreSumm/temp \
#    -train_batch_size 1500 \
#    -test_batch_size 1500 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path ckgg/bert_data_noScenario/  \
#    -model_path ckgg/models_copysent_graph_noScenario/ \
#    -result_path ckgg/results_copysent_graph_noScenario/ \
#    -log_file ckgg/logs/log_copysent_graph_noScenario \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

###########################
#python train_graph.py \
#    -split_gen True \
#    -copy_decoder True \
#    -copy_word True \
#    -gpu 4 \
#    -valid_step 500 \
#    -do_eval True \
#    -train_steps 20000 \
#    -save_checkpoint_steps 10000 \
#    -temp_dir /home/ychen/PreSumm/temp \
#    -train_batch_size 1500 \
#    -test_batch_size 1500 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path image_data_20210118/bert_data_0.3_7/ \
#    -model_path image_data_20210118/models_copyword/ \
#    -result_path image_data_20210118/results_copyword/ \
#    -log_file image_data_20210118/logs/log_copyword \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

#python train_graph.py \
#    -split_qm True \
#    -copy_decoder True \
#    -copy_word True \
#    -encode_q True \
#    -graph_transformer True \
#    -train_from image_data_20210118/models_copyword_graph/model_step_14000.pt \
#    -use_cls True \
#    -avg_sent True \
#    -gpu 4 \
#    -valid_step 500 \
#    -do_eval True \
#    -train_steps 20000 \
#    -save_checkpoint_steps 10000 \
#    -temp_dir /home/ychen/PreSumm/temp \
#    -train_batch_size 1500 \
#    -test_batch_size 1500 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path image_data_20210118/bert_data_0.3_7/  \
#    -model_path image_data_20210118/models_copyword_graph/ \
#    -result_path image_data_20210118/results_copyword_graph/ \
#    -log_file image_data_20210118/logs/log_copyword_graph \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

#python train_graph.py \
#    -split_qm True \
#    -copy_decoder True \
#    -copy_word True \
#    -encode_q True \
#    -graph_transformer True \
#    -use_cls True \
#    -avg_sent True \
#    -copy_sent True \
#    -gpu 1 \
#    -valid_step 500 \
#    -do_eval True \
#    -train_steps 20000 \
#    -save_checkpoint_steps 10000 \
#    -temp_dir /home/ychen/PreSumm/temp \
#    -train_batch_size 1500 \
#    -test_batch_size 1500 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path image_data_20210118/bert_data_onlyScenario/  \
#    -model_path image_data_20210118/models_onlyScenario/ \
#    -result_path image_data_20210118/results_onlyScenario/ \
#    -log_file image_data_20210118/logs/log_onlyScenario \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

#python train_graph.py \
#    -split_qm True \
#    -copy_decoder True \
#    -copy_word True \
#    -encode_q True \
#    -graph_transformer False \
#    -train_from image_data_20210118/models_copysent/model_step_15500.pt \
#    -use_cls True \
#    -avg_sent True \
#    -copy_sent True \
#    -gpu 2 \
#    -valid_step 500 \
#    -do_eval True \
#    -train_steps 20000 \
#    -save_checkpoint_steps 10000 \
#    -temp_dir /home/ychen/PreSumm/temp \
#    -train_batch_size 1500 \
#    -test_batch_size 1500 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path image_data_20210118/bert_data_0.3_7/  \
#    -model_path image_data_20210118/models_copysent/ \
#    -result_path image_data_20210118/results_copysent/ \
#    -log_file image_data_20210118/logs/log_copysent \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

#python train_graph.py \
#    -split_qm True \
#    -copy_decoder True \
#    -copy_word True \
#    -encode_q True \
#    -mode dev \
#    -test_from image_data_20210118/models_copysent_graph/model_step_17000.pt \
#    -graph_transformer True \
#    -use_cls True \
#    -avg_sent True \
#    -copy_sent True \
#    -gpu 3 \
#    -valid_step 500 \
#    -do_eval True \
#    -train_steps 20000 \
#    -save_checkpoint_steps 10000 \
#    -temp_dir /home/ychen/PreSumm/temp \
#    -train_batch_size 1500 \
#    -test_batch_size 1500 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path image_data_20210118/bert_data_0.3_7/  \
#    -model_path image_data_20210118/models_copysent_graph/ \
#    -result_path image_data_20210118/results_copysent_graph/ \
#    -log_file image_data_20210118/logs/log_copysent_graph \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

python train_graph.py \
    -split_qm True \
    -copy_decoder True \
    -copy_word True \
    -encode_q True \
    -mode train \
    -graph_transformer True \
    -use_cls True \
    -avg_sent True \
    -copy_sent True \
    -gpu 2 \
    -valid_step 500 \
    -do_eval True \
    -train_steps 20000 \
    -save_checkpoint_steps 10000 \
    -temp_dir /home/home1/ychen/geography/temp \
    -train_batch_size 1500 \
    -test_batch_size 1500 \
    -sep_optim true \
    -lr_bert 0.0002 \
    -lr_dec 0.002 \
    -warmup_steps_bert 1000 \
    -warmup_steps_dec 1000 \
    -data_path ../data/  \
    -model_path /data/ychen/geo/github/ \
    -result_path ../data/results/ \
    -log_file ../data/log_copysent_graph \
    -rnn_hidden_size 256 \
    -rnn_num_layers 2 \
    -rel_dim 100 \
    -ff_embed_dim 1024 \
    -num_heads 8\
    -snt_layers 1\
    -graph_layers 4\
    -inference_layers 3\

#python train_graph.py \
#    -split_qm True \
#    -copy_decoder True \
#    -copy_word True \
#    -encode_q True \
#    -graph_transformer True \
#    -use_cls True \
#    -avg_sent True \
#    -copy_sent True \
#    -gpu 3 \
#    -valid_step 500 \
#    -do_eval True \
#    -train_steps 20000 \
#    -save_checkpoint_steps 20000 \
#    -temp_dir /home/ychen/PreSumm/temp \
#    -train_batch_size 1500 \
#    -test_batch_size 1500 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path image_data_20210118/bert_data_0.3_5/  \
#    -model_path image_data_20210118/models_copysent_graph_5/ \
#    -result_path image_data_20210118/results_copysent_graph_5/ \
#    -log_file image_data_20210118/logs/log_copysent_graph_5 \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

#python train_graph.py \
#    -split_qm True \
#    -copy_decoder True \
#    -copy_word True \
#    -encode_q True \
#    -split_gen True \
#    -sent_attn True \
#    -gpu 3 \
#    -valid_step 500 \
#    -temp_dir /home/ychen/PreSumm/temp \
#    -batch_size 500 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path graph_data/ \
#    -model_path graph_data/models_splitgen/ \
#    -result_path graph_data/results_splitgen/ \
#    -log_file graph_data/logs_splitgen/log \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

#python train_graph.py \
#    -split_gen True \
#    -train_steps 20000 \
#    -gpu 1 \
#    -valid_step 500 \
#    -temp_dir /home/ychen/PreSumm/temp \
#    -train_batch_size 8 \
#    -test_batch_size 1 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path graph_data/ \
#    -model_path graph_data/models_splitgen/ \
#    -result_path graph_data/results_splitgen/ \
#    -log_file graph_data/logs_splitgen/log \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

#python train_graph.py \
#    -split_qm True \
#    -copy_decoder True \
#    -copy_word True \
#    -encode_q True \
#    -graph_transformer False \
#    -use_cls True \
#    -avg_sent True \
#    -comfirm_connect True \
#    -train_steps 20000 \
#    -save_checkpoint_steps 5000 \
#    -gpu 2 \
#    -valid_step 500 \
#    -temp_dir  /home/ychen/PreSumm/temp \
#    -train_batch_size 1500 \
#    -test_batch_size 1500 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path image_data_20201226/bert_data/ \
#    -model_path image_data_20201226/model_copysent/ \
#    -result_path image_data_20201226/results_copysent/ \
#    -log_file image_data_20201226/logs/log_copysent \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3\

#python train_graph.py \
#    -split_qm True \
#    -copy_decoder True \
#    -copy_word True \
#    -comfirm_connect True \
#    -use_cls True \
#    -avg_sent True \
#    -graph_transformer True \
#    -train_steps 40000 \
#    -save_checkpoint_steps 5000 \
#    -train_from image_data_20201227/model/model_step_20000_joint.pt \
#    -classify True \
#    -multidoc True \
#    -gpu 3 \
#    -valid_step 500 \
#    -temp_dir  /home/ychen/PreSumm/temp \
#    -train_batch_size 1000 \
#    -test_batch_size 1000 \
#    -sep_optim true \
#    -lr_bert 0.0002 \
#    -lr_dec 0.002 \
#    -warmup_steps_bert 1000 \
#    -warmup_steps_dec 1000 \
#    -data_path image_data_20201227/bert_data_multidoc/ \
#    -model_path image_data_20201227/model1/ \
#    -result_path image_data_20201227/results1/ \
#    -log_file image_data_20201227/logs/log1 \
#    -rnn_hidden_size 256 \
#    -rnn_num_layers 2 \
#    -rel_dim 100 \
#    -ff_embed_dim 1024 \
#    -num_heads 8\
#    -snt_layers 1\
#    -graph_layers 4\
#    -inference_layers 3