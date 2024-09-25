# -*- coding: utf-8 -*-


import numpy as np
import random
random.seed(209)

#   order(0) topic(1) document(2) event1_mid(3) event2_mid(4) event_type1(5)
#   event_type2(6) event_mention1(7) event_mention2(8) label(9)
#   sentence1(10) sentence1_id(11) sentence2(12) sentence2_id(13) event_id1(14) event_id2(15)

def load_data(args):
    data_document = np.load('data/MAVENSubWoRe.npy', allow_pickle=True).item()
    train_data, valid_data, test_data = data_document['train'], data_document['valid'], data_document['test']
    sample_train_data = fewShot(args, train_data)
    return sample_train_data, valid_data, test_data


def processNode(data_doc):
    proNode = {}
    for t in data_doc:
        docList = []
        for i in range(len(data_doc[t])):
            node, candiSet = [], []
            for e in data_doc[t][i]['node']:
                node.append((e[0], e[1], e[2], e[3], e[4], e[5][0][2], e[5][0][3], e[5][0][4], e[5][0][5]))     # 选择第一个共指事件
            for e in data_doc[t][i]['candiSet']:
                candiSet.append((e[0], e[1], e[2], e[3], e[4], e[5][0][2], e[5][0][3], e[5][0][4], e[5][0][5]))
            docList.append({'node': node, 'edge': data_doc[t][i]['edge'], 'adja': data_doc[t][i]['adja'], 'candiSet': candiSet, 'label': data_doc[t][i]['label']})
        proNode[t] = docList
    return proNode

def fewShot(args,data):
    sampleData = random.sample(data, int(len(data) * args.Sample_rate))
    return sampleData



# train_data, valid_data, test_data=load_data(1,'MAVENPad.npy')
# train_data2, valid_data2, test_data2=load_data(1)
# print(111)