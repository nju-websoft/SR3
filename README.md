# SR3

SR3: Sentence Ranking, Reasoning, and Replication for Scenario-Based Essay Question Answering

## Environment

- python 3.6
- ubuntu 18.04
- cuda 11.3

## Packages

- ltp==4.0.9
- torch==1.4.0
- pytorch_transformers==1.1.0
- pyrouge

## Quickstart

### step1: sentence retrieve

```bash
cd data_process
python sentence_retrieve.py
```

### step2: sentence ranking

```bash
cd ERNIE
bash script/rank.sh
cd ../data_process
python generate_SEQA_data.py
```

environment and pretrained model of ERNIE could be installed and download from [link](https://github.com/PaddlePaddle/ERNIE).

sentence ranking model after training could be download from [link](https://drive.google.com/file/d/1noA5I7jlqglSrfjb1GsWy7qcHtGF6QXc/view?usp=sharing).

### step3: answer generation

```bash
cd SEQA
bash run_graph.sh
```

#### detail of run_graph.sh

```bash
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
```
answer generation model after training could be download from [link](https://drive.google.com/file/d/11VKNSdH9NxerKcZf3Tql26-qa1-XcghD/view?usp=sharing).

arguments:

| argument          | description                        |
| ----------------- | ---------------------------------- |
| mode              | could be "train", "dev", "test"    |
| copy_word         | if or not copy in word level       |
| graph_transformer | if or not use graph transformer    |
| copy_sent         | if or not copy in sentence level   |
| do_eval           | do evaluation while training       |
| valid_step        | do evaluation every ? steps        |
| temp_dir          | directory to save pretrained model |
| data_path         | directory of data                  |
| model_path        | directory to save model            |
| results_path      | directory to save results          |
| log_file          | path to save log                   |
| test_from         | path of model to load              |


