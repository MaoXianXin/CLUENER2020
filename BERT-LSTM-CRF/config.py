import os
import torch

data_dir = os.getcwd() + '/data/clue/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
bert_model = 'pretrained_bert_models/bert-base-chinese/'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
model_dir = os.getcwd() + '/experiments/clue/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 32
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = '1'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

labels = ['address', 'book', 'company', 'game', 'government',
          'movie', 'name', 'organization', 'position', 'scene', 'tech']

label2id = {
    "O": 0,
    "B-address": 1,
    "B-book": 2,
    "B-company": 3,
    'B-game': 4,
    'B-government': 5,
    'B-movie': 6,
    'B-name': 7,
    'B-organization': 8,
    'B-position': 9,
    'B-scene': 10,
    'B-tech': 11,
    "I-address": 12,
    "I-book": 13,
    "I-company": 14,
    'I-game': 15,
    'I-government': 16,
    'I-movie': 17,
    'I-name': 18,
    'I-organization': 19,
    'I-position': 20,
    'I-scene': 21,
    'I-tech': 22,
    "S-address": 23,
    "S-book": 24,
    "S-company": 25,
    'S-game': 26,
    'S-government': 27,
    'S-movie': 28,
    'S-name': 29,
    'S-organization': 30,
    'S-position': 31,
    'S-scene': 32,
    'S-tech': 33
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
