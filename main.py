# -*- coding: utf-8 -*-

# This project is for Roberta model.

import time
import random
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
import torch.nn as nn
import logging
import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from load_data import load_data
from transformers import RobertaTokenizer, AdamW
from parameter import parse_args
from util import correct_data, replace_mult_event, load_json_from_file, savePredict,collect_mult_event
from tools import calculate, get_batch
import random

from model import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.cuda.empty_cache()
args = parse_args()  # load parameters

if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.model):
    os.mkdir(args.model)
if not os.path.exists(args.train_file):
    os.mkdir(args.train_file)
if not os.path.exists(args.dev_file):
    os.mkdir(args.dev_file)
if not os.path.exists(args.test_file):
    os.mkdir(args.test_file)
t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + 'maven' + '__' + t + '.txt'
args.model = args.model + 'maven' + '__' + t + '.pth'

# refine
for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')

logger = logging.getLogger(__name__)


def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)

# set seed for random number
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setup_seed(args.seed)

# load Roberta model
printlog('Passed args:')
printlog('log path: {}'.format(args.log))
printlog('transformer model: {}'.format(args.model_name))

tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

# load data tsv file
printlog('Loading data')

train_data, dev_data, test_data = load_data(args)
train_size = len(train_data)
dev_size = len(dev_data)
test_size = len(test_data)
print('Data loaded')


train_data=correct_data(train_data)
dev_data=correct_data(dev_data)
test_data=correct_data(test_data)



# multi_event,special_multi_event_token,event_dict,reverse_event_dict,to_add=collect_mult_event(train_data+dev_data+test_data,tokenizer)
reverse_event_dict = load_json_from_file("data/reverse_event_dict.json")
to_add = load_json_from_file("data/to_add.json")
special_multi_event_token = [reverse_event_dict[k] for k in reverse_event_dict.keys()]


# additional_mask=['<v1>','<c>','<c2>','</c>','</c2>','<d>','</d>']     #50265、50266、50267、50268、50269、50270、50271
# tokenizer.add_tokens(additional_mask)           #7
tokenizer.add_tokens(special_multi_event_token) #516
args.vocab_size = len(tokenizer)                #50265+7+516


train_data = replace_mult_event(train_data,reverse_event_dict)
dev_data = replace_mult_event(dev_data,reverse_event_dict)
test_data = replace_mult_event(test_data,reverse_event_dict)



# ---------- network ----------

net = MLP(args).to(device)
net.handler(to_add, tokenizer)


no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.t_lr)


cross_entropy = nn.CrossEntropyLoss().to(device)

# save model and result
best_mrr, best_hit1, best_hit3, best_hit10, best_hit20, best_hit50= 0, 0, 0, 0, 0, 0
dev_best_mrr, dev_best_hit1, dev_best_hit3, dev_best_hit10, dev_best_hit20, dev_best_hit50 = 0, 0, 0, 0, 0, 0
best_mrr_epoch, best_hit1_epoch, best_hit3_epoch, best_hit10_epoch, best_hit20_epoch, best_hit50_epoch = 0, 0, 0, 0, 0, 0
state = {}

best_epoch = 0

printlog(args)
printlog('Start training ...')


##################################  epoch  #################################
for epoch in range(args.num_epoch):
    args.model = './outmodel/epoch' + str(epoch) + '__' + t + '.pth'
    print('=' * 20)
    printlog('Epoch: {}'.format(epoch))
    torch.cuda.empty_cache()

    all_indices = torch.randperm(train_size).split(args.batch_size)
    loss_epoch = 0.0

    Mrr, Hit1, Hit3, Hit10, Hit20, Hit50 = [], [], [], [], [], []

    all_Mrr, all_Hit1, all_Hit3, all_Hit10, all_Hit20, all_Hit50 = [], [], [], [], [], []

    start = time.time()

    printlog('lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
    printlog('t_lr:{}'.format(optimizer.state_dict()['param_groups'][1]['lr']))

    train_predict_file = open(args.train_file + 'train_file_' + str(epoch) + '__' + t + '.txt', "w")
    dev_predict_file = open(args.dev_file + 'dev_file_' + str(epoch) + '__' + t + '.txt', "w")
    test_predict_file = open(args.test_file + 'test_file_' + str(epoch) + '__' + t + '.txt', "w")

    ############################################################################
    ##################################  train  #################################
    ############################################################################
    net.train()
    progress = tqdm.tqdm(total=len(train_data) // args.batch_size + 1, ncols=75,
                         desc='Train {}'.format(epoch))
    total_step = len(train_data) // args.batch_size + 1
    step = 0
    for ii, batch_indices in enumerate(all_indices, 1):
        mode = 'SimPrompt Learning'
        progress.update(1)
        # get a batch of wordvecs
        batch_arg, mask_arg, batch_Type_arg, mask_Type_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, labels, candiSet = get_batch(train_data, args, batch_indices, tokenizer)

        candiLabels = [] +labels
        for tt in range(len(labels)):
            candiLabels[tt] = candiSet[tt].index(labels[tt])
        batch_arg, mask_arg = batch_arg.to(device), mask_arg.to(device)
        batch_Type_arg, mask_Type_arg = batch_Type_arg.to(device), mask_Type_arg.to(device)
        mask_indices = mask_indices.to(device)
        for sent in sentences:
            for k in sent.keys():
                sent[k]['input_ids'] = sent[k]['input_ids'].to(device)
                sent[k]['attention_mask'] = sent[k]['attention_mask'].to(device)
        length = len(batch_indices)
        # fed data into network
        prediction, SP_loss = net(mode, batch_arg, mask_arg, batch_Type_arg, mask_Type_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, candiSet, candiLabels, length)

        # answer_space：[23702,50265]
        label = torch.LongTensor(labels).to(device)
        # loss
        loss = cross_entropy(prediction,label) + args.Sim_ratio * SP_loss

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        loss_epoch += loss.item()

        savePredict(batch_indices, train_predict_file, prediction, candiSet, labels)

        mrr, hit1, hit3, hit10, hit20, hit50 = calculate(prediction, candiSet, labels, length)
        Mrr += mrr
        Hit1 += hit1
        Hit3 += hit3
        Hit10 += hit10
        Hit20 += hit20
        Hit50 += hit50

        all_Mrr += mrr
        all_Hit1 += hit1
        all_Hit3 += hit3
        all_Hit10 += hit10
        all_Hit20 += hit20
        all_Hit50 += hit50

        if ii % (500 // args.batch_size) == 0:
            printlog('loss={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit20={:.4f} hit50={:.4f}'.format(
                loss_epoch / (30 // args.batch_size),
                sum(Hit1) / len(Hit1),
                sum(Hit3) / len(Hit3),
                sum(Hit10) / len(Hit10),
                sum(Hit20) / len(Hit20),
                sum(Hit50) / len(Hit50)))
            loss_epoch = 0.0
            Mrr, Hit1, Hit3, Hit10, Hit20, Hit50 = [], [], [], [], [], []
    end = time.time()
    progress.close()

    ############################################################################
    ##################################  dev  ###################################
    ############################################################################
    all_indices = torch.range(0,dev_size-1,dtype=torch.int32).split(args.batch_size)
    mode = 'Prompt Learning'
    Mrr_d, Hit1_d, Hit3_d, Hit10_d, Hit20_d, Hit50_d = [], [], [], [], [], []

    progress = tqdm.tqdm(total=len(dev_data) // args.batch_size + 1, ncols=75,
                         desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # get a batch of dev_data
        batch_arg, mask_arg, batch_Type_arg, mask_Type_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, labels, candiSet = get_batch(dev_data, args, batch_indices, tokenizer)

        candiLabels = [] + labels
        for tt in range(len(labels)):
            candiLabels[tt] = candiSet[tt].index(labels[tt])
        batch_arg, mask_arg = batch_arg.to(device), mask_arg.to(device)
        batch_Type_arg, mask_Type_arg = batch_Type_arg.to(device), mask_Type_arg.to(device)
        mask_indices = mask_indices.to(device)
        for sent in sentences:
            for k in sent.keys():
                sent[k]['input_ids'] = sent[k]['input_ids'].to(device)
                sent[k]['attention_mask'] = sent[k]['attention_mask'].to(device)
        length = len(batch_indices)
        # fed data into network
        prediction = net(mode, batch_arg, mask_arg, batch_Type_arg, mask_Type_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, candiSet, candiLabels, length)

        savePredict(batch_indices, dev_predict_file, prediction, candiSet, labels)

        mrr, hit1, hit3, hit10, hit20, hit50 = calculate(prediction, candiSet, labels, length)
        Mrr_d += mrr
        Hit1_d += hit1
        Hit3_d += hit3
        Hit10_d += hit10
        Hit20_d += hit20
        Hit50_d += hit50

    progress.close()

    ############################################################################
    ##################################  test  ##################################
    ############################################################################
    all_indices = torch.range(0,test_size-1,dtype=torch.int32).split(args.batch_size)
    mode = 'Prompt Learning'
    Mrr_t, Hit1_t, Hit3_t, Hit10_t, Hit20_t, Hit50_t = [], [], [], [], [], []

    progress = tqdm.tqdm(total=len(test_data) // args.batch_size + 1, ncols=75,
                         desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # get a batch of dev_data
        batch_arg, mask_arg, batch_Type_arg, mask_Type_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, labels, candiSet = get_batch(test_data, args, batch_indices, tokenizer)

        candiLabels = [] + labels
        for tt in range(len(labels)):
            candiLabels[tt] = candiSet[tt].index(labels[tt])
        batch_arg, mask_arg = batch_arg.to(device), mask_arg.to(device)
        batch_Type_arg, mask_Type_arg = batch_Type_arg.to(device), mask_Type_arg.to(device)
        mask_indices = mask_indices.to(device)
        for sent in sentences:
            for k in sent.keys():
                sent[k]['input_ids'] = sent[k]['input_ids'].to(device)
                sent[k]['attention_mask'] = sent[k]['attention_mask'].to(device)
        length = len(batch_indices)
        # fed data into network
        prediction = net(mode, batch_arg, mask_arg, batch_Type_arg, mask_Type_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, candiSet, candiLabels, length)

        savePredict(batch_indices, test_predict_file, prediction, candiSet, labels)

        mrr, hit1, hit3, hit10, hit20, hit50 = calculate(prediction, candiSet, labels, length)
        Mrr_t += mrr
        Hit1_t += hit1
        Hit3_t += hit3
        Hit10_t += hit10
        Hit20_t += hit20
        Hit50_t += hit50

    progress.close()

    ############################################################################
    ##################################  result  ##################################
    ############################################################################
    ######### Train Results Print #########
    printlog('-------------------')
    printlog("TIME: {}".format(time.time() - start))
    printlog('EPOCH : {}'.format(epoch))
    printlog("TRAIN:")
    printlog('loss={:.4f} mrr={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit2-={:.4f} hit50={:.4f}'.format(
        loss_epoch / (30 // args.batch_size),
        sum(all_Mrr) / len(all_Mrr),
        sum(all_Hit1) / len(all_Hit1),
        sum(all_Hit3) / len(all_Hit3),
        sum(all_Hit10) / len(all_Hit10),
        sum(all_Hit20) / len(all_Hit20),
        sum(all_Hit50) / len(all_Hit50)))

    ######### Dev Results Print #########
    printlog("DEV:")
    printlog('loss={:.4f} mrr={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit2-={:.4f} hit50={:.4f}'.format(
        loss_epoch / (30 // args.batch_size),
        sum(Mrr_d) / len(Mrr_d),
        sum(Hit1_d) / len(Hit1_d),
        sum(Hit3_d) / len(Hit3_d),
        sum(Hit10_d) / len(Hit10_d),
        sum(Hit20_d) / len(Hit20_d),
        sum(Hit50_d) / len(Hit50_d)))

    ######### Test Results Print #########
    printlog("TEST:")
    printlog('loss={:.4f} mrr={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit2-={:.4f} hit50={:.4f}'.format(
        loss_epoch / (30 // args.batch_size),
        sum(Mrr_t) / len(Mrr_t),
        sum(Hit1_t) / len(Hit1_t),
        sum(Hit3_t) / len(Hit3_t),
        sum(Hit10_t) / len(Hit10_t),
        sum(Hit20_t) / len(Hit20_t),
        sum(Hit50_t) / len(Hit50_t)))

    # record the best result
    if sum(Mrr_d) / len(Mrr_d) > dev_best_mrr:
        dev_best_mrr = sum(Mrr_d) / len(Mrr_d)
        best_mrr = sum(Mrr_t) / len(Mrr_t)
        best_mrr_epoch = epoch
    if sum(Hit1_d) / len(Hit1_d) > dev_best_hit1:
        dev_best_hit1 = sum(Hit1_d) / len(Hit1_d)
        best_hit1 = sum(Hit1_t) / len(Hit1_t)
        best_hit1_epoch = epoch
    if sum(Hit3_d) / len(Hit3_d) > dev_best_hit3:
        dev_best_hit3 = sum(Hit3_d) / len(Hit3_d)
        best_hit3 = sum(Hit3_t) / len(Hit3_t)
        best_hit3_epoch = epoch
    if sum(Hit10_d) / len(Hit10_d) > dev_best_hit10:
        dev_best_hit10 = sum(Hit10_d) / len(Hit10_d)
        best_hit10 = sum(Hit10_t) / len(Hit10_t)
        best_hit10_epoch = epoch
    if sum(Hit20_d) / len(Hit20_d) > dev_best_hit20:
        dev_best_hit20 = sum(Hit20_d) / len(Hit20_d)
        best_hit20 = sum(Hit20_t) / len(Hit20_t)
        best_hit20_epoch = epoch
    if sum(Hit50_d) / len(Hit50_d) > dev_best_hit50:
        dev_best_hit50 = sum(Hit50_d) / len(Hit50_d)
        best_hit50 = sum(Hit50_t) / len(Hit50_t)
        best_hit50_epoch = epoch

    printlog('=' * 20)
    printlog('Best result at mrr epoch: {}'.format(best_mrr_epoch))
    printlog('Best result at hit1 epoch: {}'.format(best_hit1_epoch))
    printlog('Best result at hit3 epoch: {}'.format(best_hit3_epoch))
    printlog('Best result at hit10 epoch: {}'.format(best_hit10_epoch))
    printlog('Best result at hit20 epoch: {}'.format(best_hit20_epoch))
    printlog('Best result at hit50 epoch: {}'.format(best_hit50_epoch))
    printlog('Eval mrr: {}'.format(best_mrr))
    printlog('Eval hit1: {}'.format(best_hit1))
    printlog('Eval hit3: {}'.format(best_hit3))
    printlog('Eval hit10: {}'.format(best_hit10))
    printlog('Eval hit20: {}'.format(best_hit20))
    printlog('Eval hit50: {}'.format(best_hit50))

    train_predict_file.close()
    dev_predict_file.close()
    test_predict_file.close()