import torch
import torch

seqs = torch.zeros(0, 0, 0)

print(seqs)
#print(len(seqs))

def create_empty_batch():
    def __init__(self):
        self.seqs = seqs
        self.targets = targets
        self.seq_lengths = seq_lengths
        self.target_lengths = target_lengths


    seqs = torch.zeros(0, 0, 0)
    targets = torch.zeros(0, 0).to(torch.long)
    seq_lengths = list()
    target_lengths = list()
    return seqs, targets, seq_lengths, target_lengths

batch = create_empty_batch()
print(type(batch[0]))
print(type(batch[0].size(0)))
#
# def seq_length_(p):
#     return len(p[0])
#
# def target_length_(p):
#     return len(p[1])
#
# seq_lengths = [len(s[0]) for s in batch]
# target_lengths = [len(s[1]) for s in batch]
#
# max_seq_sample = max(batch, key=seq_length_)[0]
#
#
#
# print(batch.seq_length.shape)

