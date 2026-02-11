'''
This script handles the training process.
Updated for modern PyTorch/torchtext versions.
'''

import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

__author__ = "Yu-Hsiang Huang"


class TranslationDataset(Dataset):
    """PyTorch Dataset for translation task"""
    
    def __init__(self, examples, src_vocab, trg_vocab):
        self.examples = examples
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert tokens to indices
        # Add BOS and EOS tokens
        # 获取原序列的token，并将其转换到vocab的索引，添加BOS+EOS
        src_indices = [Constants.BOS] + [
            self.src_vocab.stoi.get(tok, Constants.UNK) 
            for tok in example.src
        ] + [Constants.EOS]
        
        trg_indices = [Constants.BOS] + [
            self.trg_vocab.stoi.get(tok, Constants.UNK) 
            for tok in example.trg
        ] + [Constants.EOS]
        
        return {
            'src': torch.LongTensor(src_indices), #[Das, ist, ein, Hund] => 转换为词典索引 [2, 45, 32, 18, 67, 3]
            'trg': torch.LongTensor(trg_indices) #[This, is, a, dog] => 转换为词典索引 [2, 45, 32, 18, 67, 3]
        }

# batch中每个样本和白标签对都是以map的形式存在的 [{'src': xxx, 'trg': yyy}, {'src': xxx, 'trg': yyy}]
# 将基于map形式的batch转换为tensor形式的batch
def collate_fn(batch, src_pad_idx, trg_pad_idx):
    """Collate function for DataLoader - pads sequences to same length"""
    src_seqs = [item['src'] for item in batch]
    trg_seqs = [item['trg'] for item in batch]
    
    # Pad sequences
    # 填充序列，找到一个batch中最大的序列长度，作为batch的序列长度，然后填充
    src_max_len = max(len(s) for s in src_seqs)
    trg_max_len = max(len(t) for t in trg_seqs)
    
    # 填充序列
    src_padded = torch.full((len(batch), src_max_len), src_pad_idx, dtype=torch.long)
    trg_padded = torch.full((len(batch), trg_max_len), trg_pad_idx, dtype=torch.long)
    
    for i, (src, trg) in enumerate(zip(src_seqs, trg_seqs)):
        src_padded[i, :len(src)] = src
        trg_padded[i, :len(trg)] = trg
    
    return src_padded, trg_padded


class Batch:
    """Simple batch class to mimic old torchtext behavior"""
    def __init__(self, src, trg):
        # Transpose to [seq_len, batch_size] to match old format
        self.src = src.transpose(0, 1)
        self.trg = trg.transpose(0, 1)


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src

# 目标序列拆分，通过目标序列得出：Decoder输入和最终标签
def patch_trg(trg, pad_idx):
    """
        转换目标序列格式,并分离输入和标签
        
        这是 Teacher Forcing 的关键: 目标序列右移一位
        
        输入:
            trg: [trg_len, batch_size] - 完整目标序列,包含 <BOS> 和 <EOS>
            
        输出:
            trg: [batch_size, trg_len-1]     - 解码器输入 (去掉最后一个 token)
            gold: [batch_size * (trg_len-1)] - 标签 (去掉第一个 token,展平)
            
        示例:
            原始 trg (转置后): [[<BOS>, w1, w2, w3, <EOS>],
                            [<BOS>, w4, w5, <PAD>, <PAD>]]
                            
            处理后:
            trg  (解码器输入，右移一位): [[<BOS>, w1, w2, w3],
                            [<BOS>, w4, w5, <PAD>]]
                            
            gold (真实标签，左移一位并展开):   [w1, w2, w3, <EOS>, w4, w5, <PAD>, <PAD>]
                            (展平的一维向量)
    """
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, gold, opt.trg_pad_idx, smoothing=smoothing) 
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100*valid_accu))

        if opt.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)


def main():
    ''' 
    Usage:
    python train_new.py -data_pkl sample_data.pkl -output_dir output -b 8 -epoch 10
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=8)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise ValueError("Please specify -output_dir")

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')
    print(f'[Info] Using device: {device}')

    #========= Loading Dataset =========#
    if opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise ValueError("Please specify -data_pkl")

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    print(f'[Info] Source vocabulary size: {opt.src_vocab_size}')
    print(f'[Info] Target vocabulary size: {opt.trg_vocab_size}')

    # Check vocab sharing
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To share word embedding, the src/trg word2idx table shall be the same.'

    src_vocab = data['vocab']['src'].vocab
    trg_vocab = data['vocab']['trg'].vocab

    # Create PyTorch datasets
    train_dataset = TranslationDataset(data['train'], src_vocab, trg_vocab)
    val_dataset = TranslationDataset(data['valid'], src_vocab, trg_vocab)

    print(f'[Info] Training examples: {len(train_dataset)}')
    print(f'[Info] Validation examples: {len(val_dataset)}')

    # Create collate function with padding indices
    def collate_wrapper(batch):
        src_padded, trg_padded = collate_fn(batch, opt.src_pad_idx, opt.trg_pad_idx)
        return Batch(src_padded, trg_padded)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_wrapper,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_wrapper,
        num_workers=0
    )

    return train_loader, val_loader


if __name__ == '__main__':
    main()
