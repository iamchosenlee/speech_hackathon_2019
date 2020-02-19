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
import numpy as np
import librosa
import label_loader
from spec_augment_pytorch import spec_augment

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000
min_level_db = -100

target_dict = dict()

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            #44_0531_818_0_04570_02
            #531 621 625 662 292 624 662 412 334 192

            target_dict[key] = target

def _normalize(S) :
    return np.clip((S - min_level_db)/ -min_level_db, 0, 1)



def aug_get_spectrogram_feature(filepath):
    """
    (rate, width, sig) = wavio.readwav(filepath)
    #sig, sample_rate = librosa.core.load(filepath, 16000)
    sig = sig.ravel()


    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.030*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)

    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5);
    amag = stft.numpy();
    feat = torch.FloatTensor(amag)
    feat = torch.FloatTensor(feat).transpose(0, 1)
    """
    """
    input_nfft = int(round(sample_rate * 0.025))
    input_stride = int(round(sample_rate * 0.010))
    #S = np.abs(librosa.stft(sig))
    #mel_spec = librosa.feature.melspectrogram(sr=sample_rate, y=sig, n_mels=40, n_fft=512, hop_length=128)
    #mel_spec = librosa.feature.melspectrogram(sr=sample_rate, y=sig, n_mels=128, n_fft=N_FFT, win_length=int(0.030*SAMPLE_RATE),hop_length=int(0.01*SAMPLE_RATE))
    mel_spec = librosa.feature.melspectrogram(sr=sample_rate, y=sig, n_fft=2048, hop_length=512)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    #mel_spec = _normalize(mel_spec)
    #mel_spec = torch.FloatTensor(mel_spec)
    mel_spec = torch.FloatTensor(mel_spec).transpose(0,1)
    """

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


    """
    sample_rate = 16000
    hop_length = 128

    sig, sample_rate = librosa.core.load(filepath, sample_rate)

    mfcc_feat = librosa.feature.mfcc(y=sig, sr=sample_rate, hop_length = hop_length, n_mfcc = 257,n_fft=512)
    mfcc_feat = torch.FloatTensor(mfcc_feat).transpose(0,1)
    """

    return mel_spectrogram

def get_spectrogram_feature(filepath):
    sample_rate = 16000
    hop_length = 128

    sig, sample_rate = librosa.core.load(filepath, sample_rate)

    mel_spectrogram = librosa.feature.melspectrogram(y=sig, n_mels=128, sr=sample_rate, n_fft=512,hop_length=128)

    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = _normalize(mel_spectrogram)

    mel_spectrogram = torch.FloatTensor(mel_spectrogram).transpose(0,1)

    return mel_spectrogram

def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
                result.append(int(tokens[i]))
    result.append(eos_id)
    return result


"""
def get_script_rev(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    #numlist = list() #숫자인 항목 append
    isNum = False
    numLengthlist = list()
    numLength = 0
    #명 시 월 일
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            temp = int(tokens[i])
            if(temp == (123 or 249 or 0 or 65 or 87 or 132 or 200 or 249 or 285 or 306 or 435 or 488 or 646 or 674 or 677 or 712 or 722 or 729 or 738 or 745)) :
                result.append(662) #일단 대체로 빈칸을 집어넣는다
            else :
                if (temp == (151 or 214 or 455 or 72 or 590 or 675 or 88 or 110 or 552 or 569)) : #만약 숫자라면
                    if(isNum != True) :
                        isNum = True
                    #numlist.append(i)
                    numLength += 1
                else :
                    if (isNum) :
                        isNum = False
                        numLengthlist.append(numLength)
                        numLength = 0
                result.append(temp)
    result.append(eos_id)

    result = change_script(result, numLengthlist)
    return result


def change_script(result, numLengthList) :
    char2index, index2char = label_loader.load_label('./hackathon.labels')
    num_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_index = [151, 214, 455, 72, 590, 675, 88, 110, 552, 569]
    #[char2index[_] for _ in [str(_) for _ in range(10)]]

    for i in range(len(numLengthList)) :
        for j in range(len(result)) :
            if (result[j] == 2) : #명시월일 일 경우에
                #숫자인것을 한글로 바꿔줘야한다
                number = ""
                for k in range(numLengthList[i]) : #숫자의 수만큼 앞으로가서 token을 읽어온다
                    for x in num_index :
                        if (numLengthList[j + k - numLengthList[x]] == x) :
                            number += x
                            break
                    #token을 숫자로 바꿔주기
                tokenNum = int(number)
                newList= list()
                if result[j] == char2index['명'] :
                    newList += '명'
                elif (result[j] == char2index['시']) :
                    newList += '명'
                elif result[j] == char2index['분'] :
                    newList += '명'
                elif result[j] == char2index['월']:
                    newList += '명'
                elif result[j] == char2index['일']:
                    newList += '명'

                #number읽는법을 변환
                #token가져오기
                #result 숫자 범위없애고 결과 token 삽입

    return result
"""


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
                if i%2 == 0 :
                    items.append(self.dataset.auggetitem(self.index))
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

