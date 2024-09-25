import torch
from numpy import random
from util import getDistance


def getTemplate(args, data):
    edge = data['edge'][:-1] if len(data['edge'])<=(args.len_arg)//10 else data['edge'][0:(args.len_arg)//10]
    # random.shuffle(edge)
    causeRel = edge[0:len(edge)]
    template, templateType = '', ''
    relation = [] + causeRel
    assert data['edge'][-1] not in relation

    distance = getDistance(relation+[data['edge'][-1]])
    assert len(relation)+1==len(distance)

    weighted_characters = list(zip(distance[:-1], relation))
    sorted_characters = sorted(weighted_characters, reverse=True)
    sorted_relation_only = [char for weight, char in sorted_characters]


    # random.shuffle(relation)
    for rel in sorted_relation_only:
        eId1 = rel[0]
        eId2 = rel[-1]
        rl = data['node'][eId1][5] + ' ' + rel[1] + ' ' + data['node'][eId2][5]
        rlType = data['node'][eId1][4] + ' '+rel[1]+' ' + data['node'][eId2][4]
        template = template + rl + ' , '
        templateType = templateType + rlType + ' , '
    maskRel = data['edge'][-1]
    template = template + data['node'][maskRel[0]][5] + ' ' + maskRel[1] + ' <mask> .'
    templateType=templateType+data['node'][maskRel[0]][4] + ' '+maskRel[1]+' <mask> .'
    assert len(template.split(' ')) == len(templateType.split(' '))
    return template, templateType, sorted_relation_only + [maskRel]

def getSentence(args, tokenizer, data, relation):
    sentence = {}
    for rel in relation:
        if rel[0] not in sentence.keys():
            sentence[rel[0]] = data['node'][rel[0]][6]
        if rel[-1] not in sentence.keys():
            sentence[rel[-1]] = data['node'][rel[-1]][6]
    sentTokenizer = {}
    for e in sentence.keys():
        sent_dict = tokenizer.encode_plus(
                sentence[e],
                add_special_tokens=True,
                padding='max_length',
                max_length=args.len_arg,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        event = data['node'][e][5]
        sentTokenizer[str(e)+'_'+str(tokenizer.encode(event)[1])] = {'input_ids':      sent_dict['input_ids'],
                                                                     'attention_mask': sent_dict['attention_mask'],
                                                                     'position':       torch.where(sent_dict['input_ids']==tokenizer.encode(event)[1])[1].item()}
    return sentTokenizer


def tokenizerHandler(args, template, tokenizer):
    encode_dict = tokenizer.encode_plus(
        template,
        add_special_tokens=True,
        padding='max_length',
        max_length=args.len_arg,
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    arg_1_idx = encode_dict['input_ids']
    arg_1_mask = encode_dict['attention_mask']
    return arg_1_idx, arg_1_mask

def getposHandler(data, arg_idx, relation, sentence, tokenizer):
    tempPosition = torch.nonzero(arg_idx >= tokenizer.encode('<a_0>')[1]).tolist()
    ePosition = [row[-1] for row in tempPosition]
    ePositionKey = []
    sentId = []
    for rel in relation:
        sentId.append(rel[0])
        sentId.append(rel[-1])
    assert len(sentId) - 1 == len(ePosition)
    for iid in range(len(ePosition)):
        event = data['node'][sentId[iid]][5]
        ePositionKey.append(str(sentId[iid]) + '_' + str(arg_idx[0][ePosition[iid]].item()))
    return ePosition, ePositionKey

# tokenize sentence and get event idx
def get_batch(data, args, indices, tokenizer):
    batch_idx, batch_mask = [], []
    batch_Type_idx, batch_Type_mask = [], []
    event_tokenizer_pos, event_key_pos = [], []
    mask_indices, sentences, labels, candiSet= [],[],[],[]
    for idx in indices:
        candi = [tokenizer.encode(data[idx]['candiSet'][i][5])[1] for i in range(len(data[idx]['candiSet']))]
        template, templateType, relation = getTemplate(args, data[idx])
        sentence = getSentence(args, tokenizer, data[idx], relation)

        arg_idx, arg_mask = tokenizerHandler(args, template, tokenizer)
        arg_Type_idx, arg_Type_mask = tokenizerHandler(args, templateType, tokenizer)
        assert arg_mask.tolist() == arg_Type_mask.tolist()
        assert relation[-1] == data[idx]['edge'][-1]
        assert candi[data[idx]['label']] == tokenizer.encode(data[idx]['node'][relation[-1][-1]][5])[1]
        label = tokenizer.encode(data[idx]['node'][relation[-1][-1]][5])[1]
        labels.append(label)
        # template分词后所有事件的位置
        ePosition, ePositionKey = getposHandler(data[idx], arg_idx, relation, sentence, tokenizer)
        eTypePosition, eTypePositionKey = getposHandler(data[idx], arg_Type_idx, relation, sentence, tokenizer)
        assert ePosition == eTypePosition
        event_tokenizer_pos.append(ePosition)
        event_key_pos.append(ePositionKey)
        sentences.append(sentence)
        candiSet.append(candi)
        if len(batch_idx) == 0:
            batch_idx, batch_mask = arg_idx, arg_mask
            batch_Type_idx, batch_Type_mask = arg_Type_idx, arg_Type_mask
            mask_indices = torch.nonzero(arg_idx == 50264, as_tuple=False)[0][1]
            mask_indices = torch.unsqueeze(mask_indices, 0)
        else:
            batch_idx, batch_mask = torch.cat((batch_idx, arg_idx), dim=0), torch.cat((batch_mask, arg_mask), dim=0)
            batch_Type_idx, batch_Type_mask = torch.cat((batch_Type_idx, arg_Type_idx), dim=0), torch.cat((batch_Type_mask, arg_Type_mask), dim=0)
            mask_indices = torch.cat((mask_indices, torch.unsqueeze(torch.nonzero(arg_idx == 50264, as_tuple=False)[0][1], 0)), dim=0)
    return batch_idx, batch_mask, batch_Type_idx, batch_Type_mask, event_tokenizer_pos, event_key_pos, mask_indices, sentences, labels, candiSet


# calculate p, r, f1
def calculate(prediction, candiSet, labels, batch_size):
    mrr, hit1, hit3, hit10, hit20, hit50 = [], [], [], [], [], []
    for i in range(batch_size):
        predtCandi = prediction[i][candiSet[i]].tolist()
        label = candiSet[i].index(labels[i])
        labelScore = predtCandi[label]
        predtCandi.sort(reverse=True)
        rank = predtCandi.index(labelScore)
        mrr.append(1/(rank+1))
        hit1.append(int(rank<1))
        hit3.append(int(rank<3))
        hit10.append(int(rank<10))
        hit20.append(int(rank < 20))
        hit50.append(int(rank < 50))

    return mrr, hit1, hit3, hit10, hit20, hit50

