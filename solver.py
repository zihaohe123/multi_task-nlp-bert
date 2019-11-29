import torch
import torch.nn as nn
import os
from model import MultiTaskBert
from dataset import data_loader
from utils import prepar_data
from transformers import AdamW
import time
from utils import get_current_time, calc_eplased_time_since, to_device
from apex import amp



class Solver:
    def __init__(self, args):
        # how to use GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_workers = max([4 * torch.cuda.device_count(), 4])

        prepar_data()

        # prepare data
        train_loader, dev_loader, test_loader = data_loader(
            path=args.data_path, batch_size=args.batch_size,
            num_workers=num_workers,
            pin_memory=device == 'cuda')
        print('#examples:',
              '#train', len(train_loader.dataset),
              '#dev', len(dev_loader.dataset),
              '#test', len(test_loader.dataset))

        model = MultiTaskBert(args)
        model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=5e-4)

        if args.apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        device_count = 0
        if device == 'cuda':
            device_count = torch.cuda.device_count()
            if device_count > 1:
                model = nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True
            print("Let's use {} GPUs!".format(device_count))

        criterion_classification = nn.CrossEntropyLoss()
        criterion_regression = nn.MSELoss()

        name = 'multi_task_bert' if args.multi_task else 'single_task_bert'
        ckpt_path = os.path.join('ckpt', '{}.pth'.format(name))

        batches = len(train_loader.dataset) // args.batch_size
        log_interval = batches // 30

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion_classification = criterion_classification
        self.criterion_regression = criterion_regression
        self.device = device
        self.ckpt_path = ckpt_path
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.log_interval = log_interval

    def train(self):
        print('Starting Traing....')
        best_loss = float('inf')
        best_acc = 0.
        best_epoch = 0

        train_start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            print('-'*20 + 'Epoch: {}, {}'.format(epoch, get_current_time()) + '-'*20)
            train_loss, train_acc = self.train_epoch()
            dev_loss, dev_acc = self.evaluate_epoch('Dev')
            if dev_loss < best_loss:
                best_loss = dev_loss
                best_acc = dev_acc
                best_epoch = epoch
                self.save_model()

            print('Epoch: {:0>2d}/{}\n'
                  'Epoch Training Time: {}\n'
                  'Elapsed Time: {}\n'
                  'Train Loss: {:.3f}, Train Acc: {:.3f}\n'
                  'Dev Loss: {:.3f}, Dev Acc: {:.3f}\n'
                  'Best Dev Loss: {:.3f}, Best Dev Acc: {:.3f}, '
                  'Best Dev Acc Epoch: {:0>2d}\n'.format(epoch, self.args.epochs,
                                                       calc_eplased_time_since(epoch_start_time),
                                                       calc_eplased_time_since(train_start_time),
                                                       train_loss, train_acc,
                                                       dev_loss, dev_acc,
                                                       best_loss, best_acc, best_epoch))

            # LSTM learning rate decay
            for param_group in self.optimizer.param_groups:
                print('lr: {:.6f} -> {:.6f}\n'.format(param_group['lr'], param_group['lr'] * self.args.lr_decay))
                param_group['lr'] *= self.args.lr_decay

        print('Training Finished!')

        self.test()

    def test(self):
        # Load the best checkpoint
        self.load_model()

        # Test
        print('Final result..............')
        test_loss, test_acc = self.evaluate_epoch('Test')
        print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(test_loss, test_acc))

    def train_epoch(self):
        self.model.train()
        train_loss = 0.
        example_count = 0
        correct = 0
        batch_start_time = time.time()
        for batch_idx, (sst2_token_ids, sst2_mask_ids, sst2_labels,
                        stsb_token_ids, stsb_seg_ids, stsb_mask_ids, stsb_labels,
                        qnli_token_ids, qnli_seg_ids, qnli_mask_ids, qnli_labels,
                        snli_token_ids, snli_seg_ids, snli_mask_ids, snli_labels) in enumerate(self.train_loader):

            output = self.model(snli_token_ids, snli_seg_ids, snli_mask_ids,
                                sst2_token_ids, sst2_mask_ids,
                                stsb_token_ids, stsb_seg_ids, stsb_mask_ids,
                                qnli_token_ids, qnli_seg_ids, qnli_mask_ids)

            snli_output, sst2_output, stsb_output, qnli_output = output

            snli_labels, sst2_labels, stsb_labels, qnli_labels = \
                to_device(snli_labels, sst2_labels, stsb_labels, qnli_labels, device=self.device)
            self.optimizer.zero_grad()
            snli_loss = self.criterion_classification(snli_output, snli_labels)
            sst2_loss = self.criterion_classification(sst2_output, sst2_labels) if self.args.multi_task else 0
            stsb_loss = self.criterion_regression(stsb_output, stsb_labels) if self.args.multi_task else 0
            qnli_loss = self.criterion_classification(qnli_output, qnli_labels) if self.args.multi_task else 0

            loss = snli_loss + sst2_loss + stsb_loss + qnli_loss

            if self.args.grad_max_norm > 0.:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_max_norm)

            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()

            batch_size = snli_token_ids.shape[0]
            train_loss += batch_size * loss.item()
            example_count += batch_size

            pred = torch.max(snli_output, 1)[1]
            correct += pred.eq(snli_labels.view_as(pred)).sum().item()

            if batch_idx == 0 or (batch_idx+1) % self.log_interval == 0 or batch_idx+1 == self.log_interval:
                print('Batch: {:0>5d}/{:0>5d}, '
                      'Batch Training Time: {}, '
                      'Batch Loss: {:.3f}, '
                      'Batch SNLI Loss: {:.3f}, '
                      'Batch SST-2 Loss: {:.3f}, '
                      'Batch STS-B Loss: {:.3f}, '
                      'Batch QNLI Loss: {:.3f}, '.format(batch_idx+1, len(self.train_loader),
                                                  calc_eplased_time_since(batch_start_time),
                                                  loss, snli_loss, sst2_loss, stsb_loss, qnli_loss))
                batch_start_time = time.time()

        train_loss /= len(self.train_loader.dataset)
        acc = correct / len(self.train_loader.dataset)
        print()
        return train_loss, acc

    def evaluate_epoch(self, mode):
        print('Evaluating....')
        self.model.eval()
        if mode == 'Dev':
            loader = self.dev_loader
        else:
            loader = self.test_loader
        eval_loss = 0.
        correct = 0
        with torch.no_grad():
            for batch_idx, (snli_token_ids, snli_seg_ids, snli_mask_ids, snli_labels) in enumerate(loader):
                output, _, _, _ = self.model(snli_token_ids, snli_seg_ids, snli_mask_ids)
                snli_labels = to_device(snli_labels, device=self.device)
                loss = self.criterion_classification(output, snli_labels)
                eval_loss += len(output) * loss.item()
                pred = torch.max(output, 1)[1]
                correct += pred.eq(snli_labels.view_as(pred)).sum().item()
        eval_loss /= len(loader.dataset)
        acc = correct / len(loader.dataset)
        return eval_loss, acc

    def save_model(self):
        model_dict = dict()
        model_dict['state_dict'] = self.model.state_dict()
        model_dict['m_config'] = self.args
        model_dict['optimizer'] = self.optimizer.state_dict()
        if not os.path.exists(os.path.dirname(self.ckpt_path)):
            os.makedirs(os.path.dirname(self.ckpt_path))
        torch.save(model_dict, self.ckpt_path)
        print('Saved', self.ckpt_path)
        print()

    def load_model(self):
        print('Load checkpoint', self.ckpt_path)
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            # if saving a paralleled model but loading an unparalleled model
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(checkpoint['state_dict'])