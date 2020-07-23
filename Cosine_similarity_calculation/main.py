import torch
import torch.nn as nn
import time
import os
import sys
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser("glove.6B.300d cosine similarity")
parser.add_argument('--data_path', type=str, default='./glove.6B/glove.6B.300d.txt', help='location of the glove data')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--K', type=int, default=100, help='top K similar words')
parser.add_argument('--result_path', type=str, default='result.txt', help='location of the result data')
args = parser.parse_args()


def indices2words(indices_list):
    words_str = "[ "
    for _, _index in enumerate(indices_list):
        words_str += id2word[_index.item()]
        words_str += " "  # 分隔符
    words_str += " ]\n"
    return words_str


start_time = time.time()
data = []
id2word = defaultdict(str)
word2id = defaultdict(int)
if not torch.cuda.is_available():
    print("error: not found gpu")
    exit(0)

cudnn.benchmark = True
cudnn.enabled = True

with open(args.data_path, "r") as f:
    print("data processing...")
    index = 0
    for line in tqdm(f.readlines()):
        dt = line.split(' ')
        value = []
        for i in range(1, len(dt)):
            value.append(float(dt[i]))
        data.append(value)
        word2id[dt[0]] = index
        id2word[index] = dt[0]
        index += 1

print('data processing done!')
print('-' * 89)
data = torch.Tensor(data).cuda()
print('using gpu..., device id:' + str(data.device))
cos = nn.CosineSimilarity()

print("computing cosine similarity...")
similar_words_id = []
for i in tqdm(range(len(id2word))):
    vector = data[i:i + 1]
    result = cos(vector, data)
    _, indices = torch.topk(result, k=(args.K + 1))
    similar_words_id.append(indices[1:])

print("cosine similarity computing finish!")
print('-' * 89)
print("write data to disk...")

with open(args.result_path, "w") as f:
    for i in tqdm(range(len(id2word))):
        indices = similar_words_id[i]
        words = str(id2word[i]) + " : "
        words += indices2words(indices)
        f.writelines(words)
print("finish !")
print("time consuming : " + str((time.time() - start_time) / 60) + " minutes")
