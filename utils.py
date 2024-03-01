__author__ = 'jindi'

import collections
from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import os
import numpy as np
import random
import torch.distributed as dist
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score, precision_score, recall_score, \
    classification_report
import logging
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


# encode the sequence length information in the batch for RNN use
# this is special for pytorch RNN function
def prepare_rnn_seq(rnn_input, lengths, hx=None, masks=None, batch_first=False):
    '''

    Args:
        rnn_input: [seq_len, batch, input_size]: tensor containing the features of the input sequence.
        lengths: [batch]: tensor containing the lengthes of the input sequence
        hx: [num_layers * num_directions, batch, hidden_size]: tensor containing the initial hidden state for each element in the batch.
        masks: [seq_len, batch]: tensor containing the mask for each element in the batch.
        batch_first: If True, then the input and output tensors are provided as [batch, seq_len, feature].

    Returns:

    '''

    def check_decreasing(lengths):
        lens, order = torch.sort(lengths, dim=0, descending=True)
        if torch.ne(lens, lengths).sum() == 0:
            return None
        else:
            _, rev_order = torch.sort(order)
            return lens, order, rev_order

    check_res = check_decreasing(lengths)

    if check_res is None:
        lens = lengths
        rev_order = None
    else:
        lens, order, rev_order = check_res
        batch_dim = 0 if batch_first else 1
        rnn_input = rnn_input.index_select(batch_dim, order)
        if hx is not None:
            # hack lstm
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx.index_select(1, order)
                cx = cx.index_select(1, order)
                hx = (hx, cx)
            else:
                hx = hx.index_select(1, order)

    lens = lens.tolist()
    seq = rnn_utils.pack_padded_sequence(rnn_input, lens, batch_first=batch_first)
    if masks is not None:
        if batch_first:
            masks = masks[:, :lens[0]]
        else:
            masks = masks[:lens[0]]
    return seq, hx, rev_order, masks


# recover the sequence results from RNN function
# this is special to pytorch RNN function
def recover_rnn_seq(seq, rev_order, hx=None, batch_first=False):
    output, _ = rnn_utils.pad_packed_sequence(seq, batch_first=batch_first)
    if rev_order is not None:
        batch_dim = 0 if batch_first else 1
        output = output.index_select(batch_dim, rev_order)
        if hx is not None:
            # hack lstm
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx.index_select(1, rev_order)
                cx = cx.index_select(1, rev_order)
                hx = (hx, cx)
            else:
                hx = hx.index_select(1, rev_order)
    return output, hx


def process_text(text):
    text = text.split(' ')
    new_text = []
    i = 0
    while i < len(text):
        tmp = text[i]
        if not text[i].startswith('##'):
            j = i + 1
            # print(text[i])
            # print(j ,text[j], text[j].startswith('##'))
            while j < len(text) and text[j].startswith('##'):
                tmp += text[j][2:]
                # print(text[j], tmp)
                j += 1
            i = j
        else:
            i += 1
        new_text.append(tmp)

    return ' '.join(new_text)


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    random.seed(seed)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    torch.cuda.set_device(args.gpu)


def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    # logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger


def accuracy(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, zero_division=0)
    leng = len(labels)
    count = 0

    for i in range(leng):
        if labels[i] == preds[i]:
            count += 1

    acc = count / leng
    print('%f' % acc)

    acc = '%f' % acc
    p = precision_score(labels, preds, average='macro', zero_division=0)
    r = recall_score(labels, preds, average='macro', zero_division=0)
    f = f1_score(labels, preds, average='macro', zero_division=0)

    # return acc, precision, recall, f1, p, r, f
    return acc, p, r, f, p, r, f


def create_optimizer(model, lr_pretrained=1e-5, lr_random=5e-5, no_decay_names=[], pretrained_names=[],
                     warmup_steps=1000, max_steps=100000, schedule_type='warmup'):
    weight_decay = 0.01

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_names)
                   and any(bb in n for bb in pretrained_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr_pretrained,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_names)
                   and any(bb in n for bb in pretrained_names)
            ],
            "weight_decay": 0.0,
            "lr": lr_pretrained,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_names)
                   and not any(bb in n for bb in pretrained_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr_random,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_names)
                   and not any(bb in n for bb in pretrained_names)
            ],
            "weight_decay": 0.0,
            "lr": lr_random,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=lr_random, eps=1e-8, betas=(0.9, 0.999)
    )
    # optimizer.param_groups

    if isinstance(warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)
    print('warmup steps: {} | max steps: {}'.format(warmup_steps, max_steps))

    if schedule_type == 'warmup_linear-decay':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=max_steps)
    elif schedule_type == 'warmup_poly-decay':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=0,
            power=1,
        )
    elif schedule_type == 'warmup':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    else:
        scheduler = None

    return optimizer, scheduler


@torch.no_grad()
def momentum_update(model_pair, momentum):
    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        param_m.data = param_m.data * momentum + param.data * (1. - momentum)


@torch.no_grad()
def copy_params(model_pair):
    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        param_m.data.copy_(param.data)
        param_m.requires_grad = False

