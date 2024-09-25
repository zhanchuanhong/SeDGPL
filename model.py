# coding: UTF-8
import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM, BertForMaskedLM
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
embedding_size = 768


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.roberta_model = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
        self.roberta_model.resize_token_embeddings(args.vocab_size)
        for param in self.roberta_model.parameters():
            param.requires_grad = True

        self.robert_text = deepcopy(self.roberta_model)
        self.robert_type = deepcopy(self.roberta_model)

        for param in self.robert_text.parameters():
            param.requires_grad = True

        self.hidden_size = 768

        # gate1
        self.W1_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W1_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # gate2
        self.W2_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W2_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.vocab_size = args.vocab_size

    # batch_arg:句子分词id，arg_mask:句子分词掩码，mask_indices:[MASK]在分词id中的位置，event_group:事件id集合
    def forward(self, mode, batch_arg, arg_mask, batch_Type_arg, mask_Type_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, candiSet, candiLabels, batch_size):

        for i in range(batch_size):
            for k in sentences[i]:
                sent_emb = self.robert_text.roberta(sentences[i][k]['input_ids'], sentences[i][k]['attention_mask'])[0].to(device)
                sentences[i][k]['emb'] = sent_emb[0][sentences[i][k]['position']]

        Type_emb = self.robert_type.roberta(batch_Type_arg, attention_mask=mask_Type_arg, output_hidden_states=True)[0].to(device)

        word_emb = self.roberta_model.roberta.embeddings.word_embeddings(batch_arg).to(device)

        for i in range(batch_size):
            for j in range(len(event_tokenizer_pos[i])):
                instance_emb = (word_emb[i][event_tokenizer_pos[i][j]]).clone().unsqueeze(0)
                sent_emb = (sentences[i][event_key_pos[i][j]]['emb']).clone().unsqueeze(0)
                type_emb = (Type_emb[i][event_tokenizer_pos[i][j]]).clone().unsqueeze(0)
                gate_1 = torch.sigmoid(self.W1_1(instance_emb) + self.W1_2(sent_emb)).to(device)
                out_gate_1 = (torch.mul(gate_1, instance_emb) + torch.mul(1.0 - gate_1, sent_emb)).to(device)

                gate_2 = torch.sigmoid(self.W2_1(out_gate_1) + self.W2_2(type_emb)).to(device)
                out_gate_2 = (torch.mul(gate_2, out_gate_1) + torch.mul(1.0 - gate_2, type_emb)).to(device).squeeze(0)

                word_emb[i][event_tokenizer_pos[i][j]] = out_gate_2
                assert str(int(batch_arg[i][event_tokenizer_pos[i][j]])) in event_key_pos[i][j]

        temp_emb = self.roberta_model.roberta(attention_mask = arg_mask, inputs_embeds = word_emb)[0].to(device)

        anchor_maks = torch.tensor([]).to(device)
        for i in range(batch_size):
            e_emb = self.extract_event(temp_emb[i], mask_indices[i])
            if i == 0:
                anchor_maks = e_emb
            else:
                anchor_maks = torch.cat((anchor_maks, e_emb),dim=0)
        if mode == 'Prompt Learning':
            prediction = self.roberta_model.lm_head(anchor_maks)
            return prediction
        if mode == 'SimPrompt Learning':
            prediction = self.roberta_model.lm_head(anchor_maks)
            candi_e_emb = self.roberta_model.roberta.embeddings.word_embeddings(torch.tensor(candiSet[0]).to(device)).to(device)
            pos_emb = candi_e_emb[candiLabels, :]
            neg_emb = torch.cat((candi_e_emb[:candiLabels[0]], candi_e_emb[candiLabels[0] + 1:]))

            pos_cos = F.cosine_similarity(anchor_maks, pos_emb, dim=1)
            neg_cos = F.cosine_similarity(anchor_maks, neg_emb, dim=1)

            pos_sim = torch.sum(torch.exp(pos_cos), dim=0)
            neg_sim = torch.sum(torch.exp(neg_cos), dim=0)

            SP_loss = - torch.log(pos_sim / (pos_sim + neg_sim))

            return prediction, SP_loss


    def extract_event(self, embed, mask_idx):
        mask_embed = embed[mask_idx]
        mask_embed = torch.unsqueeze(mask_embed, 0)
        return mask_embed
    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        da = self.roberta_model.roberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp
