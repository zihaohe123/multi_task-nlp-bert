import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pickle
from transformers import BertTokenizer


class GLUEData:
    def __init__(self, path, name, tokenizer, params, label_dict=None):
        self.path = path
        self.name = name
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.params = params

        if name != 'SNLI':
            self.train_data_path = os.path.join(path, name, 'train.tsv')
            self.dev_data_path = os.path.join(path, name, 'dev.tsv')
            self.test_data_path = None
        else:
            self.train_data_path = os.path.join(path, name, 'train.txt')
            self.dev_data_path = os.path.join(path, name, 'dev.txt')
            self.test_data_path = os.path.join(path, name, 'test.txt')

    def get_data(self, phase):
        print('Preparing {} {} data....'.format(self.name, phase))
        pkl_data_path = os.path.join(self.path, self.name, '{}.pkl'.format(phase))
        if os.path.exists(pkl_data_path):
            print('Found {} {} data'.format(self.name, phase))
            with open(pkl_data_path, 'rb') as f:
                return pickle.load(f)
        else:
            data = self.load_data(getattr(self, '{}_data_path'.format(phase)))
            with open(pkl_data_path, 'wb') as f:
                pickle.dump(data, f)
            return data

    def load_data(self, path):
        if len(self.params) == 2:
            return self.load_data1(path, self.params[0], self.params[1])
        else:
            return self.load_data2(path, self.params[0], self.params[1], self.params[2])

    def load_data1(self, path, seq_col, label_col):
        token_ids_ = []
        token_lens = []
        labels = []

        with open(path, 'r', newline='', encoding='utf-8') as f:
            for idx, line in enumerate(f):

                # skip the first line
                if idx == 0:
                    continue
                if idx % 5000 == 0:
                    print(idx)

                cols = line.strip('\n').split('\t')

                seq = cols[seq_col]
                label = cols[label_col]

                #   '–' indicates a lack of consensus from the human annotators, ignore it
                if label == '-':
                    continue

                label = self.label_dict[label] if self.label_dict else label

                tokens = self.tokenizer.tokenize(seq)

                # the maximum input length of BERT base model is 512
                if len(tokens) > 510:
                    tokens = tokens[:510]
                tokens = ['[CLS]'] + tokens + ['[SEP]']
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                token_len = len(token_ids)

                token_ids_.append(torch.tensor(token_ids))
                token_lens.append(token_len)
                labels.append(label)

        token_ids_ = pad_sequence(token_ids_, batch_first=True)
        token_lens = torch.tensor(token_lens)
        labels = torch.tensor(labels)

        return token_ids_, token_lens, labels

    def load_data2(self, path, seq1_col, seq2_col, label_col):
        pair_token_ids_ = []
        seq1_lens = []
        seq2_lens = []
        labels = []

        with open(path, 'r', newline='', encoding='utf-8') as f:
            for idx, line in enumerate(f):

                # skip the first line
                if idx == 0:
                    continue
                if idx % 5000 == 0:
                    print(idx)

                cols = line.strip('\n').split('\t')

                seq1, seq2 = cols[seq1_col], cols[seq2_col]
                label = cols[label_col]

                #   '–' indicates a lack of consensus from the human annotators, ignore it
                if label == '-':
                    continue
                label = float(label) if label[0].isdigit() else label
                label = self.label_dict[label] if self.label_dict else label

                tokens1, tokens2 = self.tokenizer.tokenize(seq1), self.tokenizer.tokenize(seq2)
                # the maximum input length of BERT base model is 512
                if len(tokens1) > 254:
                    tokens1 = tokens1[:254]
                if len(tokens2) > 255:
                    tokens2 = tokens2[:255]
                tokens1 = ['[CLS]'] + tokens1 + ['[SEP]']
                tokens2 = tokens2 + ['[SEP]']

                token_ids1, token_ids2 = self.tokenizer.convert_tokens_to_ids(tokens1), self.tokenizer.convert_tokens_to_ids(tokens2)

                seq_len1, seq_len2 = len(token_ids1), len(token_ids2)
                pair_token_ids = token_ids1 + token_ids2

                pair_token_ids_.append(torch.tensor(pair_token_ids))
                seq1_lens.append(seq_len1)
                seq2_lens.append(seq_len2)
                labels.append(label)

        pair_token_ids_ = pad_sequence(pair_token_ids_, batch_first=True)
        seq1_lens = torch.tensor(seq1_lens)
        seq2_lens = torch.tensor(seq2_lens)
        labels = torch.tensor(labels)

        return pair_token_ids_, seq1_lens, seq2_lens, labels


def padding_two_tensors(tensor1, tensor2):
    if tensor1.shape[1] > tensor2.shape[1]:
        gap = tensor1.shape[1] - tensor2.shape[1]
        padding = torch.zeros((tensor2.shape[0], gap), dtype=tensor2.dtype)
        tensor2 = torch.cat((tensor2, padding), dim=1)
        return tensor1, tensor2
    elif tensor1.shape[1] < tensor2.shape[1]:
        return padding_two_tensors(tensor2, tensor1)


def cat_train_dev(train_data, dev_data):
    data = []
    for i in range(len(train_data)):
        train_i, dev_i = train_data[i], dev_data[i]

        if train_i.ndim > 1:
            train_i, dev_i = padding_two_tensors(train_i, dev_i)

        data.append(torch.cat((train_i, dev_i), dim=0))
    return data


class MultiTaskDataset(Dataset):
    def __init__(self, path, phase, multi_task=False):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        snli = GLUEData(path, 'SNLI', tokenizer, [5, 6, 0], {'entailment': 0, 'contradiction': 1, 'neutral': 2})
        self.multi_task = multi_task

        if phase == 'train':
            if multi_task:
                # sst2 = GLUEData(path, 'SST-2', tokenizer, [0, 1], {'0': 0, '1': 0})
                stsb = GLUEData(path, 'STS-B', tokenizer, [-3, -2, -1])
                qnli = GLUEData(path, 'QNLI', tokenizer, [-3, -2, -1], {'entailment': 0, 'not_entailment': 1})

                # we only focus on SNLI, so we don't evaluate the performance on other three datasets
                # we just use all data for training
                # self.sst2_token_ids, self.sst2_token_lens, self.sst2_labels = \
                #     cat_train_dev(sst2.get_data('train'), sst2.get_data('dev'))
                self.stsb_token_ids, self.stsb_token_lens1, self.stsb_token_lens2, self.stsb_labels = cat_train_dev(
                    stsb.get_data('train'), stsb.get_data('dev'))
                self.qnli_token_ids, self.qnli_token_lens1, self.qnli_token_lens2, self.qnli_labels = cat_train_dev(
                    qnli.get_data('train'), qnli.get_data('dev'))

                # self.sst2_len = self.sst2_token_ids.shape[0]
                self.stsb_len = self.stsb_token_ids.shape[0]
                self.qnli_len = self.qnli_token_ids.shape[0]

                # shuffle the three datasets
                # idxes = torch.randperm(self.sst2_len)
                # self.sst2_token_ids, self.sst2_token_lens, self.sst2_labels = \
                #     self.sst2_token_ids[idxes], self.sst2_token_lens[idxes], self.sst2_labels[idxes]

                idxes = torch.randperm(self.stsb_len)
                self.stsb_token_ids, self.stsb_token_lens1, self.stsb_token_lens2, self.stsb_labels = \
                    self.stsb_token_ids[idxes], self.stsb_token_lens1[idxes], self.stsb_token_lens2[idxes], self.stsb_labels[idxes]

                idxes = torch.randperm(self.qnli_len)
                self.qnli_token_ids, self.qnli_token_lens1, self.qnli_token_lens2, self.qnli_labels = \
                    self.qnli_token_ids[idxes], self.qnli_token_lens1[idxes], self.qnli_token_lens2[idxes], self.qnli_labels[idxes]

            self.snli_token_ids, self.snli_token_lens1, self.snli_token_lens2, self.snli_labels = snli.get_data('train')

        elif phase == 'dev':
            self.snli_token_ids, self.snli_token_lens1, self.snli_token_lens2, self.snli_labels = snli.get_data('dev')

        elif phase == 'test':
            self.snli_token_ids, self.snli_token_lens1, self.snli_token_lens2, self.snli_labels = snli.get_data('test')

        self.phase = phase

    def __len__(self):
        return self.snli_token_ids.shape[0]

    def __getitem__(self, index):
        if self.phase == 'train':
            if self.multi_task:
                # because the four datasets have different lengths, we need to over-sample the small datasets to make sure
                # they have the same length with the largest dataset
                # sst2_index = index % self.sst2_len
                stsb_index = index % self.stsb_len
                qnli_index = index % self.qnli_len

                # return self.sst2_token_ids[sst2_index], self.sst2_token_lens[sst2_index], self.sst2_labels[sst2_index], \
                return self.stsb_token_ids[stsb_index], self.stsb_token_lens1[stsb_index], \
                       self.stsb_token_lens2[stsb_index], self.stsb_labels[stsb_index], \
                       self.qnli_token_ids[qnli_index], self.qnli_token_lens1[qnli_index], \
                       self.qnli_token_lens2[qnli_index], self.qnli_labels[qnli_index], \
                       self.snli_token_ids[index], self.snli_token_lens1[index], \
                       self.snli_token_lens2[index], self.snli_labels[index]

            else:
                return [None]*8 + [self.snli_token_ids[index], self.snli_token_lens1[index], self.snli_token_lens2[index], self.snli_labels[index]]

        else:
            return self.snli_token_ids[index], self.snli_token_lens1[index], self.snli_token_lens2[index], self.snli_labels[index]


def batchify_seq(batch, token_ids_idx, token_lens_idx, token_labels_idx):
    token_lens = torch.tensor([b[token_lens_idx] for b in batch], dtype=torch.long)
    max_len = torch.max(token_lens).item()
    token_ids = torch.stack([b[token_ids_idx][:max_len] for b in batch])
    mask_ids = pad_sequence([torch.ones(l.item(), dtype=torch.long) for l in token_lens], batch_first=True)
    labels = torch.tensor([b[token_labels_idx] for b in batch])
    assert token_ids.shape[1] == mask_ids.shape[1]
    return token_ids, mask_ids, labels


def batchify_seq_pair(batch, token_ids_idx, token_lens1_idx, token_lens2_idx, token_labels_idx):
    token_lens1 = torch.tensor([b[token_lens1_idx] for b in batch], dtype=torch.long)
    token_lens2 = torch.tensor([b[token_lens2_idx] for b in batch], dtype=torch.long)
    token_lens = token_lens1 + token_lens2
    max_len = torch.max(token_lens).item()
    token_ids = torch.stack([b[token_ids_idx][:max_len] for b in batch])
    seg_ids = pad_sequence([torch.cat((torch.ones(l1.item(), dtype=torch.long), torch.zeros(l2.item(), dtype=torch.long)), dim=0)
                                 for l1, l2 in zip(token_lens1, token_lens2)], batch_first=True)
    mask_ids = pad_sequence([torch.ones(l.item(), dtype=torch.long) for l in token_lens], batch_first=True)
    labels = torch.tensor([b[token_labels_idx] for b in batch])
    assert token_ids.shape[1] == mask_ids.shape[1] == seg_ids.shape[1]
    return token_ids, seg_ids, mask_ids, labels


def batchify(batch):
    snli_token_ids, snli_seg_ids, snli_mask_ids, snli_labels = batchify_seq_pair(batch, -4, -3, -2, -1)
    # train
    if len(batch[0]) != 4:

        if batch[0][0] is not None:
            # sst2_token_ids, sst2_mask_ids, sst2_labels = batchify_seq(batch, 0, 1, 2)
            stsb_token_ids, stsb_seg_ids, stsb_mask_ids, stsb_labels = batchify_seq_pair(batch, 0, 1, 2, 3)
            qnli_token_ids, qnli_seg_ids, qnli_mask_ids, qnli_labels = batchify_seq_pair(batch, 4, 5, 6, 7)

        else:
            # sst2_token_ids, sst2_mask_ids, sst2_labels = None, None, None
            stsb_token_ids, stsb_seg_ids, stsb_mask_ids, stsb_labels = None, None, None, None
            qnli_token_ids, qnli_seg_ids, qnli_mask_ids, qnli_labels = None, None, None, None

        # return sst2_token_ids, sst2_mask_ids, sst2_labels, \
        return stsb_token_ids, stsb_seg_ids, stsb_mask_ids, stsb_labels, \
               qnli_token_ids, qnli_seg_ids, qnli_mask_ids, qnli_labels, \
               snli_token_ids, snli_seg_ids, snli_mask_ids, snli_labels
    # dev and test
    else:
        return snli_token_ids, snli_seg_ids, snli_mask_ids, snli_labels


def data_loader(path, batch_size, multi_task, num_workers, pin_memory):
    train_loader = DataLoader(MultiTaskDataset(path, 'train', multi_task),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              collate_fn=batchify)

    dev_loader = DataLoader(MultiTaskDataset(path, 'dev', multi_task),
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            collate_fn=batchify)

    test_loader = DataLoader(MultiTaskDataset(path, 'test', multi_task),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             collate_fn=batchify)

    return train_loader, dev_loader, test_loader
