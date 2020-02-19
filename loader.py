"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import sys
import math
import wavio
import time
import torch
import random
import threading
import logging
from torch.utils.data import Dataset, DataLoader
from label_loader import load_label
import numpy as np
import librosa
import label_loader
from spec_augment_pytorch import spec_augment

char2index, index2char = load_label('./hackathon.labels')

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000
min_level_db = -100

target_dict = dict()

def _normalize(S) :
    return np.clip((S - min_level_db)/ -min_level_db, 0, 1)

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target

def get_spectrogram_feature(filepath):
    sample_rate = 16000
    hop_length = 128

    sig, sample_rate = librosa.core.load(filepath, sample_rate)

    mel_spectrogram = librosa.feature.melspectrogram(y=sig, n_mels=128, sr=sample_rate, n_fft=512,hop_length=128)

    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = _normalize(mel_spectrogram)

    mel_spectrogram = torch.FloatTensor(mel_spectrogram).transpose(0,1)

    return mel_spectrogram

def aug_get_spectrogram_feature(filepath):

    sample_rate = 16000
    hop_length = 128

    sig, sample_rate = librosa.core.load(filepath, sample_rate)

    mel_spectrogram = librosa.feature.melspectrogram(y=sig, n_mels=128, sr=sample_rate, n_fft=512,hop_length=128)

    shape = mel_spectrogram.shape
    mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1]))
    mel_spectrogram = torch.from_numpy(mel_spectrogram)
    mel_spectrogram = spec_augment(mel_spectrogram)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram[0,:,:], ref=np.max)
    mel_spectrogram = _normalize(mel_spectrogram)

    mel_spectrogram = torch.FloatTensor(mel_spectrogram).transpose(0,1)


    return mel_spectrogram

# def get_script(filepath, bos_id, eos_id):
#     key = filepath.split('/')[-1].split('.')[0]
#     script = target_dict[key]
#     tokens = script.split(' ')
#     result = list()
#     result.append(bos_id)
#     for i in range(len(tokens)):
#         if len(tokens[i]) > 0:
#             result.append(int(tokens[i]))
#     result.append(eos_id)
#     return result
from functools import reduce
import re
from preprocess import *
#below_five = ['417', '240', '325', '5', '202', '273', '262', '655', '216', '119', '232', '488', '514', '257', '443', '172', '555', '578', '760', '607', '445', '48', '264', '84', '358', '30', '811', '520', '193', '224', '653', '646', '287', '779', '429', '549', '35', '303', '526', '18', '197', '302', '663', '562', '61', '322', '238', '668', '156', '789', '507', '582', '782', '538', '129', '114', '57', '558', '321', '128', '722', '489', '737', '270', '639', '708', '403', '767', '801', '282', '367', '293', '13', '713', '438', '792', '762', '21', '142', '266', '671', '14', '482', '149', '568', '644', '612', '769', '599', '774', '766', '629', '535', '499', '703', '275', '472', '553', '3', '160', '473', '7', '50', '816', '204', '397', '405', '802', '435', '460', '269', '328']
not_word = ['729', '306', '132', '65', '738', '677', '488', '285', '646', '435', '745', '200', '674', '712', '662']

# ! ? 제외한 모든 특수문자 (띄어쓰기는 제거해준다)
def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')

    line = reduce(lambda x, y: x + y[0], [index2char[int(_)] for _ in tokens])
    if bool(re.search(gisu, line)):
        line = read_gisu(line)
    elif bool(re.search(seosu, line)):
        line = read_seosu(line)
    elif bool(re.search(phone, line)):
        line = read_phone(line)
    elif bool(re.search(bunji, line)):
        line = read_bunji(line)
    elif bool(re.search(r'[a-zA-Z]', line)):
        line = read_alpha(line)
    tokens = [char2index[_] for _ in line]
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if (len(tokens[i]) > 0):
                #and (tokens[i] not in below_five):
            result.append(int(tokens[i]))
        if tokens[i] in not_word:
            result.pop()
    result.append(eos_id)
    return result

# #'시' = 785, '명'= 534
# counter_index = [785, 534]
# num_index = [151, 214, 455, 72, 590, 675, 88, 110, 552, 569]
# num_2_t = ['영', '한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아혹', '열', '열하나', '열두']
# num_t_num = [[211], [130], [19], [715], [701], [669, 546], [691, 546], [624, 444], [691, 757], [274, 34], [176],
#              [176, 690, 62], [176, 19]]
#
# #ord_num : ordinal number (서수) : 일이삼사...
# #ord_2_t= ['공','일','이','삼','사','오','육','칠','팔','구']
# #월, 일, 분,
# # '-' : 87 전화번호는 보류
# ord_count = [297, 624, 355]
# ord_2_t_dict = {0: '공', 1: '일', 2: '이', 3: '삼', 4: '사', 5: '오', 6: '육', 7: '칠', 8: '팔', 9: '구', 10: '십', 30: '삼십', 100: '백'}
# ord_t_num = {0: [399], 1: [624], 2: [610], 3: [135], 4: [566], 5: [574], 6: [554], 7: [616], 8: [148], 9: [145], 10: [144], 30: [135, 144], 100: [753]}
#
# def num_to_word(int_result):
#     num_flag = False
#     raw_num_list = list()
#     int2w_result = list()
#     for i, index in enumerate(int_result):
#         int2w_result.append(index)
#         if index in num_index:
#             raw_num_list.append((i, index))
#             num_flag = True
#
#         if (index in counter_index) and (num_flag == True):
#             raw_num = int("".join([index2char[item[1]] for item in raw_num_list]))
#             #print("raw num :", raw_num)
#             if (0 <= raw_num <= 12):
#                 for _ in raw_num_list:
#                     int2w_result.remove(_[1])
#                 int2w_result.remove(index)
#                 #print(num_t_num[raw_num])
#                 int2w_result = int2w_result + num_t_num[raw_num]
#                 int2w_result.append(index)
#                 raw_num_list = list()
#                 num_flag = False
#             else:
#                 raw_num_list = list()
#                 num_flag = False
#         if (index in ord_count) and (num_flag == True):
#             raw_num = int("".join([index2char[item[1]] for item in raw_num_list]))
#             if raw_num in ord_t_num.keys():
#                 for _ in raw_num_list:
#                     int2w_result.remove(_[1])
#                 int2w_result.remove(index)
#                 # print(num_t_num[raw_num])
#                 int2w_result = int2w_result + ord_t_num[raw_num]
#                 int2w_result.append(index)
#                 raw_num_list = list()
#                 num_flag = False
#             else:
#                 raw_num_list = list()
#                 num_flag = False
#
#
#     return int2w_result


class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_spectrogram_feature(self.wav_paths[idx])
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        return feat, script

    def auggetitem(self, idx):
        feat = aug_get_spectrogram_feature(self.wav_paths[idx])
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        return feat, script

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths

class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(self.index))
                #if i%2 == 0 :
                    #items.append(self.dataset.auggetitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()

