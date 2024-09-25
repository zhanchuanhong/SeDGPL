import torch

import json
import networkx as nx
import numpy as np
from collections import deque

def load_json_from_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def isContinue(id_list):
    for i in range(len(id_list) - 1):
        if int(id_list[i]) != int(id_list[i + 1]) - 1:
            return False
    return True


def doCorrect(data):
    for i in range(len(data)):
        eId = data[i][8].split('_')[1:]
        if not isContinue(eId):
            s_1 = data[i][6].split()
            event1 = s_1[int(eId[0]):int(eId[-1]) + 1]
            event1 = ' '.join(event1)
            event1 += ' '
            new_e1_id = [str(i) for i in range(int(eId[0]), int(eId[-1]) + 1)]
            temp = ''
            for ii in new_e1_id:
                temp += s_1[int(ii)] + ' '
            assert event1 == temp
            event_place1 = '_' + '_'.join(new_e1_id)
            sentence = (
            data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], event1, data[i][6], data[i][7], event_place1)
            data[i] = sentence
    return data

def correct_data(dataSet):
    for i in range(len(dataSet)):
        dataSet[i]['node'] = doCorrect(dataSet[i]['node'])
        dataSet[i]['candiSet'] = doCorrect(dataSet[i]['candiSet'])
    return dataSet

def doCollect(data, tokenizer, multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict):
    for sentence in data:
        if sentence[5] not in multi_event:
            multi_event.append(sentence[5])
            special_multi_event_token.append("<a_" + str(len(special_multi_event_token)) + ">")
            event_dict[special_multi_event_token[-1]] = multi_event[-1]
            reverse_event_dict[multi_event[-1]] = special_multi_event_token[-1]
            to_add[special_multi_event_token[-1]] = tokenizer(multi_event[-1].strip())['input_ids'][1: -1]
    return multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict

def getEventType(data):
    eType = []
    for doc in data:
        for node in doc['node']:
            if node [4] not in eType:
                eType.append(node[4])
        for candiNode in doc['candiSet']:
            if candiNode[4] not in eType:
                eType.append(candiNode[4])
    return eType

def collect_mult_event(train_data, tokenizer):
    multi_event = []
    to_add = {}
    special_multi_event_token = []
    event_dict = {}
    reverse_event_dict = {}
    eType = getEventType(train_data)
    for sentence in train_data:
        multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict = doCollect(sentence['node'],
                                                                                                   tokenizer,
                                                                                                   multi_event, to_add,
                                                                                                   special_multi_event_token,
                                                                                                   event_dict,
                                                                                                   reverse_event_dict)
        multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict = doCollect(sentence['candiSet'],
                                                                                                   tokenizer,
                                                                                                   multi_event, to_add,
                                                                                                   special_multi_event_token,
                                                                                                   event_dict,
                                                                                                   reverse_event_dict)
    for type in eType:
        if type not in multi_event:
            multi_event.append(type)
            special_multi_event_token.append("<a_" + str(len(special_multi_event_token)) + ">")
            event_dict[special_multi_event_token[-1]] = type
            reverse_event_dict[type] = special_multi_event_token[-1]
            to_add[special_multi_event_token[-1]] = tokenizer(type.strip())['input_ids'][1: -1]
    return multi_event, special_multi_event_token, event_dict, reverse_event_dict, to_add


def doReplace(data, reverse_event_dict):
    for i in range(len(data)):
        # assert data[i][5] in reverse_event_dict
        if data[i][5] in reverse_event_dict:
            assert data[i][4] in reverse_event_dict
            sent = data[i][6].split()
            eId = data[i][8].split('_')[1:]
            eId.reverse()
            for id in eId:
                sent.pop(int(id))
            sent.insert(int(eId[-1]), reverse_event_dict[data[i][5]])
            sentence = (data[i][0], data[i][1], data[i][2], data[i][3], reverse_event_dict[data[i][4]], reverse_event_dict[data[i][5]], " ".join(sent), data[i][7], '_' + eId[-1])
            data[i]=sentence
    return data

def replace_mult_event(data, reverse_event_dict):
    for i in range(len(data)):
        data[i]['node'] = doReplace(data[i]['node'], reverse_event_dict)
        data[i]['candiSet'] = doReplace(data[i]['candiSet'], reverse_event_dict)
    return data

def savePredict(batch_indices, file, prediction, candiSet, labels):
    for i in range(len(prediction)):
        predtCandi = prediction[i][candiSet[i]].tolist()
        label = candiSet[i].index(labels[i])
        file.write(str(batch_indices[i].item()) + '\t')
        for score in predtCandi:
            file.write(str(score) + '\t')
        for id in candiSet[i]:
            file.write(str(id) + '\t')
        file.write(str(label) + '\n')
    return



def find_distances(adj_matrix, source_edge):
    graph = nx.Graph()
    num_nodes = adj_matrix.shape[0]
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))
    graph.add_edges_from(edges)

    source_node_1, source_node_2 = source_edge

    distances = {}

    queue = deque([(source_node_1, 0), (source_node_2, 0)])
    visited = set([source_node_1, source_node_2])

    while queue:
        node, distance = queue.popleft()
        distances[node] = distance

        neighbors = graph.neighbors(node)
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))
                visited.add(neighbor)
    distance_dict={}
    for edge in graph.edges():
        if edge != source_edge:
            node_1, node_2 = edge
            distance = min(distances.get(node_1, 16), distances.get(node_2, 16))
            distance_dict[edge] = distance
    return distance_dict

def getAdja(rel):
    node=[]
    edge=[]
    for r in rel:
        if r[0] not in node: node.append(r[0])
        if r[-1] not in node: node.append(r[-1])
        edge.append([r[0],r[-1]])
        edge.append([r[-1],r[0]])
    adja=np.zeros((max(node)+1, max(node)+1))
    for r in edge:
        adja[r[0],r[1]]=1
    return adja


def getDistance(relation):
    adja = getAdja(relation)
    distans_dict = find_distances(adja,(relation[-1][0], relation[-1][-1]))
    distans=[]
    for rel in relation[:-1]:
        if (rel[0], rel[-1]) in distans_dict:
            distans.append(distans_dict[(rel[0], rel[-1])])
        else:
            distans.append(distans_dict[(rel[-1], rel[0])])
    distans.append(-1)
    return distans


