# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ECI')

    # Dataset
    parser.add_argument('--len_arg', default=200, type=int, help='Sentence length')
    parser.add_argument('--len_temp', default=0, type=int, help='Template length')
    parser.add_argument('--cause_ratio', default=1, type=int, help='cause ratio')
    parser.add_argument('--becausedby_ratio', default=1, type=int, help='be caused by ratio')
    parser.add_argument('--nothing_ratio', default=1, type=int, help='nothing ratio')
    parser.add_argument('--Sim_ratio', default=0.5, type=float, help='Ratio of similarity in loss function')
    parser.add_argument('--Sample_rate', default=0.2, type=float, help='Few shot rate')

    parser.add_argument('--model_name', default='RoBERTaForMaskedLM/roberta-base', type=str, help='Model used to be encoder')
    parser.add_argument('--vocab_size', default=50265, type=int, help='Size of RoBERTa vocab')

    # Prompt and Contrastive Training
    parser.add_argument('--num_epoch', default=15, type=int, help='Number of total epochs to run prompt learning')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for prompt learning')
    parser.add_argument('--t_lr', default=5e-6, type=float, help='Initial lr')
    parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')

    # Others
    parser.add_argument('--seed', default=209, type=int, help='Seed for reproducibility')
    parser.add_argument('--log', default='./out/', type=str, help='Log result file name')
    parser.add_argument('--model', default='./outmodel/', type=str, help='Model parameters result file name')
    parser.add_argument('--train_file', default='./predict/train/', type=str, help='Log result file name')
    parser.add_argument('--dev_file', default='./predict/dev/', type=str, help='Model parameters result file name')
    parser.add_argument('--test_file', default='./predict/test/', type=str, help='Log result file name')

    args = parser.parse_args()
    return args
