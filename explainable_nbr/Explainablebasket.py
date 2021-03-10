import math

import numpy as np
import random
import sys
import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from pytorch_metric import *

class ExplainableNBRNet(nn.Module):

    def __init__(self, config, dataset):
        super(ExplainableNBRNet, self).__init__()

        # device setting
        self.device = config['device']

        # dataset features
        self.n_items = dataset['item_num']

        # model parameters
        self.embedding_size = config['embedding_size']
        self.embedding_type = config['embedding_type']
        self.hidden_size = config['hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.max_len = config['max_len'] # basket len
        self.mode = config['mode']

        self.meta_loss_tag = config['meta_loss']
        # define layers
        # self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, self.max_len)
        self.basket_embedding = Basket_Embedding(self.device, self.embedding_size, self.n_items, self.max_len, self.embedding_type)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)

        self.meta_module = Meta_Module(
            self.device,
            self.hidden_size,
            self.max_len,
            dropout_prob=self.dropout_prob,
            mode=self.mode
        )

        self.repeat_decoder = Repeat_Decoder(
            self.device,
            hidden_size=self.hidden_size,
            seq_len=self.max_len,
            num_item=self.n_items,
            dropout_prob=self.dropout_prob
        )
        if self.mode == 'single':
            self.explore_decoder = Explore_Decoder(
                self.device,
                hidden_size=self.hidden_size,
                seq_len=self.max_len,
                num_item=self.n_items,
                dropout_prob=self.dropout_prob
            )
        if self.mode == 'multiple':
            self.explore_item_decoder = Explore_Item(
                self.device,
                hidden_size=self.hidden_size,
                seq_len=self.max_len,
                num_item=self.n_items,
                dropout_prob=self.dropout_prob
            )
            self.explore_user_decoder = Explore_User(
                self.device,
                hidden_size=self.hidden_size,
                seq_len=self.max_len,
                num_item=self.n_items,
                dropout_prob=self.dropout_prob
            )
            self.explore_popular_decoder = Explore_Popular(
                self.device,
                hidden_size=self.hidden_size,
                seq_len=self.max_len,
                num_item=self.n_items,
                dropout_prob=self.dropout_prob
            )

        self.loss_fct = nn.BCELoss()
        self.meta_loss_fct = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def forward(self, basket_seq, basket_seq_len, candidates_basket):
        basket_seq_len = []
        for b in basket_seq:
            basket_seq_len.append(len(b))
        basket_seq_len = torch.as_tensor(basket_seq_len).to(self.device)

        batch_basket_seq_embed = self.basket_embedding(basket_seq)

        all_memory, _ = self.gru(batch_basket_seq_embed)
        last_memory = self.gather_indexes(all_memory, basket_seq_len-1)
        # timeline_mask = torch.ne(batch_basket_seq_embed, torch.zeros(self.embedding_size))
        # timeline_mask = (batch_basket_seq_embed == torch.zeros(self.embedding_size)) ##Note here are not settled yet.
        # print(timeline_mask.size())
        timeline_mask = get_timeline_mask(batch_basket_seq_embed, self.device, self.embedding_size)
        ## Need to get the candidates for decoder, Here we condiser the candidates could be pre-calculated.
        # including repeat_candidates, explore(exculde repeated items), explore(item, user, popular)
        repeat_candidates = candidates_basket['repeat']
        explore_candidates = candidates_basket['explore']
        explore_item_candidates = candidates_basket['item']
        explore_user_candidates = candidates_basket['user']
        explore_popular_candidates = candidates_basket['popular']
        # print(len(explore_item_candidates))

        meta_weights = self.meta_module.forward(all_memory, last_memory)
        repeat_outputs = self.repeat_decoder.forward(all_memory, last_memory, repeat_candidates, timeline_mask)

        if self.mode == 'single':
            explore_outputs = self.explore_decoder.forward(all_memory, last_memory, explore_candidates, timeline_mask)

            prediction = repeat_outputs*meta_weights[:, 0].unsqueeze(1) + explore_outputs*meta_weights[:, 1].unsqueeze(1)
            return prediction, meta_weights, repeat_outputs, explore_outputs

        if self.mode == 'multiple':
            explore_item_outputs = self.explore_item_decoder.forward(all_memory, last_memory, explore_item_candidates, timeline_mask)
            explore_user_outputs = self.explore_user_decoder.forward(all_memory, last_memory, explore_user_candidates, timeline_mask)
            explore_popular_outputs = self.explore_popular_decoder(all_memory, last_memory, explore_popular_candidates, timeline_mask)
            prediction = repeat_outputs*meta_weights[:,0].unsqueeze(1) \
                         + explore_item_outputs*meta_weights[:,1].unsqueeze(1) \
                         + explore_user_outputs*meta_weights[:,2].unsqueeze(1)\
                         + explore_popular_outputs*meta_weights[:,3].unsqueeze(1)
            return prediction, meta_weights, repeat_outputs, explore_item_outputs, explore_user_outputs, explore_popular_outputs

    def predict(self):
        pass

    def get_batch_loss(self, pred, tgt, device):
        # need to handle the case that
        batch_size = pred.size(0)
        tmp_tgt = get_label_tensor(tgt, device, self.n_items)
        loss = 0.0
        for ind in range(batch_size):
            # print(pred[ind].size(), tmp_tgt[ind].size())
            # sys.stdout.flush()
            # pred[ind][pred[ind]>1.0] = 1.0
            pred_ind = torch.clamp(pred[ind], 0.0, 0.99)
            loss += self.loss_fct(pred_ind.unsqueeze(0), tmp_tgt[ind].unsqueeze(0))
        return loss/batch_size # compute average

    def meta_loss(self, basket_seq, basket_seq_len, tgt, candidates_basket):#not sure to use the pred or tgt
        # determined by the groundtruth label distribution.If this is necessary. Need to redesign abou this part.
        # here we think about the decoder could achieve good performance (could capture the ground_truth candidates)
        batch_size = len(tgt)
        repeat_label = get_sub_label_set(tgt, candidates_basket['repeat'])
        explore_label = get_sub_label_set(tgt, candidates_basket['explore'])
        explore_item = get_sub_label_set(tgt, candidates_basket['item'])
        explore_user = get_sub_label_set(tgt, candidates_basket['user'])
        explore_popular = get_sub_label_set(tgt, candidates_basket['popular'])
        loss = 0.0
        for ind in range(batch_size):
            repeat_w = len(repeat_label[ind])/(len(tgt)*(math.log(len(candidates_basket['repeat'][ind])+2.8)))
            if self.mode == 'single':
                pred, meta_weights, repeat_pred, explore_pred = self.forward(basket_seq, basket_seq_len, candidates_basket)
                explore_w = len(explore_label[ind])/(len(tgt[ind])*len(candidates_basket['explore'][ind]))
                loss += self.meta_loss_fct(meta_weights, torch.as_tensor([repeat_w, explore_w]).to(self.device))
            if self.mode == 'multiple':
                pred, meta_weights, repeat_pred, explore_item_pred, explore_user_pred, explore_popular_pred = self.forward(
                    basket_seq, basket_seq_len, candidates_basket)
                explore_item_w = len(explore_item[ind])/(len(tgt[ind])*math.log(len(candidates_basket['item'][ind])))
                explore_user_w = len(explore_user[ind])/(len(tgt[ind])*math.log(len(candidates_basket['user'][ind])))
                explore_popular_w = len(explore_popular[ind])/(len(tgt[ind])*math.log(len(candidates_basket['popular'])))
                loss += self.meta_loss_fct(meta_weights[ind], torch.as_tensor([repeat_w, explore_item_w, explore_user_w, explore_popular_w]).to(self.device))
        return loss/batch_size


    def decoder_loss(self, basket_seq, basket_seq_len, tgt_basket, candidates_basket):
        #here we can conduct an experiment, to demonstrate this local loss would increase the exploration.
        if self.mode == 'single':
            pred, _, repeat_pred, explore_pred = self.forward(basket_seq, basket_seq_len, candidates_basket)
            repeat_loss = self.get_batch_loss(repeat_pred, get_sub_label_set(tgt_basket, candidates_basket['repeat']), self.device)
            explore_loss = self.get_batch_loss(explore_pred, get_sub_label_set(tgt_basket, candidates_basket['explore']), self.device)
            return repeat_loss, explore_loss
        if self.mode == 'multiple':
            pred, _, repeat_pred, explore_item_pred, explore_user_pred, explore_popular_pred = self.forward(basket_seq, basket_seq_len, candidates_basket)
            repeat_loss = self.get_batch_loss(repeat_pred, get_sub_label_set(tgt_basket, candidates_basket['repeat']), self.device)
            e_item_loss = self.get_batch_loss(explore_item_pred, get_sub_label_set(tgt_basket, candidates_basket['item']), self.device)
            e_user_loss = self.get_batch_loss(explore_user_pred, get_sub_label_set(tgt_basket, candidates_basket['user']), self.device)
            e_popular_loss = self.get_batch_loss(explore_popular_pred, get_sub_label_set(tgt_basket, candidates_basket['popular']), self.device)
            return repeat_loss, e_item_loss, e_user_loss, e_popular_loss

    def global_loss(self, basket_seq, basket_seq_len, tgt_basket, candidates_basket):
        if self.mode == 'single':
            prediction, _, _, _ = self.forward(basket_seq, basket_seq_len, candidates_basket)
        if self.mode == 'multiple':
            prediction, _, _, _, _, _ = self.forward(basket_seq, basket_seq_len, candidates_basket)

        loss = self.get_batch_loss(prediction+ 1e-8, tgt_basket, self.device) #the multilabel loss here
        return loss

    def calculate_loss(self, basket_seq, tgt_basket, candidates_basket):
        basket_seq_len = []
        for b in basket_seq:
            basket_seq_len.append(len(b))
        if self.mode == 'single':
            global_loss = self.global_loss(basket_seq, basket_seq_len, tgt_basket)
            repeat_loss, explore_loss = self.decoder_loss(basket_seq, basket_seq_len, tgt_basket, candidates_basket)
            meta_loss = self.meta_loss(basket_seq, basket_seq_len, tgt_basket, candidates_basket)
            return global_loss, repeat_loss, explore_loss, meta_loss
        if self.mode == 'multiple':
            global_loss = self.global_loss(basket_seq, basket_seq_len, tgt_basket, candidates_basket)
            repeat_loss, item_loss, user_loss, pop_loss = self.decoder_loss(basket_seq, basket_seq_len, tgt_basket, candidates_basket)
            # sys.stdout.flush()
            if self.meta_loss_tag:
                meta_loss = self.meta_loss(basket_seq, basket_seq_len, tgt_basket, candidates_basket)
                return global_loss, repeat_loss, item_loss, user_loss, pop_loss, meta_loss
            else:
                return global_loss, repeat_loss, item_loss, user_loss, pop_loss

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

# Provide basket embedding solution: max, mean, sum
class Basket_Embedding(nn.Module):

    def __init__(self, device, hidden_size, item_num, max_len, type): #hidden_size is the embedding_size
        super(Basket_Embedding, self).__init__()
        self.hidden_size = hidden_size
        self.n_items = item_num
        self.max_len = max_len
        self.type = type
        self.device = device
        self.item_embedding = nn.Embedding(item_num, hidden_size) #padding_idx=0, not sure???

    def forward(self, batch_basket):
        #need to padding here
        batch_embed_seq = [] # batch * seq_len * hidden size
        for basket_seq in batch_basket:
            embed_baskets = []
            for basket in basket_seq:
                basket = torch.LongTensor(basket).resize_(1, len(basket))
                basket = Variable(basket).to(self.device)
                basket = self.item_embedding(basket).squeeze(0)
                # embed_b = basket_pool(basket, 1, self.type)
                if self.type == 'mean':
                    embed_baskets.append(torch.mean(basket, 0))
                if self.type == 'max':
                    embed_baskets.append(torch.max(basket, 0)[0])
                if self.type == 'sum':
                    embed_baskets.append(torch.sum(basket, 0))
            #padding the seq
            pad_num = self.max_len -len(embed_baskets)
            for ind in range(pad_num):
                embed_baskets.append(torch.zeros(self.hidden_size).to(self.device))
            embed_seq = torch.stack(embed_baskets, 0)
            embed_seq = torch.as_tensor(embed_seq)
            batch_embed_seq.append(embed_seq)

        batch_embed_output = torch.stack(batch_embed_seq, 0).to(self.device)
        return batch_embed_output

class Meta_Module(nn.Module):
    def __init__(self, device, hidden_size, seq_len, dropout_prob, mode):
        super(Meta_Module, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.device = device
        self.seq_len = seq_len
        self.mode = mode

        # Attention Layer
        self.W_meta = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_meta = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.V_meta = nn.Linear(hidden_size, 1, bias=False)

        # Meta Layer
        if self.mode == 'single':
            self.Meta = nn.Linear(hidden_size, 2, bias=False)
        if self.mode == 'multiple':
            self.Meta = nn.Linear(hidden_size, 4)

    def forward(self, all_memory, last_memory):

        all_memory_values = all_memory
        all_memory = self.dropout(self.U_meta(all_memory))
        last_memory = self.dropout(self.W_meta(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_ere = self.tanh(all_memory + last_memory)
        output_ere = self.V_meta(output_ere)
        alpha_are = nn.Softmax(dim=1)(output_ere)
        alpha_are = alpha_are.repeat(1, 1, self.hidden_size)

        output_attention_applied = alpha_are*all_memory_values
        output_attention_applied = output_attention_applied.sum(dim=1)

        output_meta = self.dropout(self.Meta(output_attention_applied))
        return output_meta

class Repeat_Decoder(nn.Module):
    def __init__(self, device, hidden_size, seq_len, num_item, dropout_prob, update=1):
        super(Repeat_Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.device = device
        self.seq_len = seq_len
        self.n_items = num_item

        self.W_repeat = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_repeat = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()

        self.V_repeat = nn.Linear(hidden_size, 1)

        if update:
            self.Repeat = nn.Linear(hidden_size*2, num_item)
        self.sigmoid = nn.Sigmoid()

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        '''item_seq is the appared items or candidate items'''
        all_memory_values, last_memory_values = all_memory, last_memory

        all_memory = self.dropout(self.U_repeat(all_memory))
        last_memory = self.dropout(self.W_repeat(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_er = self.tanh(all_memory+last_memory)
        output_er = self.V_repeat(output_er).squeeze(-1)

        if mask is not None:
            output_er.masked_fill_(mask, -1e9)

        output_er = output_er.unsqueeze(-1)

        alpha_r = nn.Softmax(dim=1)(output_er)
        alpha_r = alpha_r.repeat(1, 1, self.hidden_size)
        output_r = (all_memory_values*alpha_r).sum(dim=1)
        output_r = torch.cat([output_r, last_memory_values], dim=1)
        output_r = self.dropout(self.Repeat(output_r))

        repeat_mask = get_candidate_mask(item_seq, self.device, self.n_items)
        output_r = output_r.masked_fill(repeat_mask.bool(), float('-inf'))
        repeat_recommendation_decoder = self.sigmoid(output_r)

        return repeat_recommendation_decoder

class Explore_Decoder(nn.Module):
    def __init__(self, device, hidden_size, seq_len, num_item, dropout_prob):
        super(Explore_Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_items = num_item
        self.device = device

        self.W_explore = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_explore = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.V_explore = nn.Linear(hidden_size, 1)

        self.Explore = nn.Linear(self.hidden_size * 2, self.n_items, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        all_memory_values, last_memory_values = all_memory, last_memory

        all_memory = self.dropout(self.U_explore(all_memory))
        last_memory = self.dropout(self.W_explore(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_ee = self.tanh(all_memory + last_memory)

        output_ee = self.V_explore(output_ee).squeeze(-1)
        if mask is not None:
            output_ee.masked_fill_(mask, -1e9)
        output_ee = output_ee.unsqueeze(-1)

        alpha_e = nn.Softmax(dim=1)(output_ee)
        alpha_e = alpha_e.repeat(1, 1, self.hidden_size)
        output_e = (alpha_e*all_memory_values).sum(dim=1)
        output_e = torch.cat([output_e, last_memory_values], dim=1)
        output_e = self.dropout(self.Explore(output_e))

        map_matrix = build_map(item_seq, self.device, max_index=self.n_items)
        explore_mask = torch.bmm((item_seq > 0).float().unsqueeze(1), map_matrix).squeeze(1)
        output_e = output_e.masked_fill(explore_mask.bool(), float('-inf')) # mask the items
        explore_recommendation_decoder = self.sigmoid(output_e)

        return explore_recommendation_decoder

class Explore_Item(nn.Module):
    def __init__(self, device, hidden_size, seq_len, num_item, dropout_prob):
        super(Explore_Item, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_items = num_item
        self.device = device

        self.U_item = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_item = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.V_item = nn.Linear(hidden_size, 1)

        self.Explore_item = nn.Linear(hidden_size*2, num_item, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        all_memory_values, last_memory_values = all_memory, last_memory

        all_memory = self.dropout(self.U_item(all_memory))
        last_memory = self.dropout(self.W_item(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_ei = self.tanh(all_memory+last_memory)

        output_ei = self.V_item(output_ei).squeeze(-1)
        if mask is not None:
            output_ei.masked_fill_(mask, -1e9)
        output_ei = output_ei.unsqueeze(-1)

        alpha_i = nn.Softmax(dim=1)(output_ei)
        alpha_i = alpha_i.repeat(1, 1, self.hidden_size)
        output_i = (all_memory_values*alpha_i).sum(dim=1)
        output_i = torch.cat([output_i, last_memory_values], dim=1)
        output_i = self.dropout(self.Explore_item(output_i))

        candidates_mask = get_candidate_mask(item_seq, self.device, self.n_items)
        output_i = output_i.masked_fill(candidates_mask.bool(), float('-inf'))
        explore_item_decoder = self.sigmoid(output_i)
        return explore_item_decoder

class Explore_User(nn.Module):
    def __init__(self, device, hidden_size, seq_len, num_item, dropout_prob):
        super(Explore_User, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_items = num_item
        self.device = device

        self.U_user = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_user = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.V_user = nn.Linear(hidden_size, 1)

        self.Explore_user = nn.Linear(hidden_size*2, num_item, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        all_memory_values, last_memory_values = all_memory, last_memory

        all_memory = self.dropout(self.U_user(all_memory))
        last_memory = self.dropout(self.W_user(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_eu = self.tanh(all_memory+last_memory)

        output_eu = self.V_user(output_eu).squeeze(-1)
        if mask is not None:
            output_eu.masked_fill_(mask, -1e9)
        output_eu = output_eu.unsqueeze(-1)

        alpha_u = nn.Softmax(dim=1)(output_eu)
        alpha_u = alpha_u.repeat(1, 1, self.hidden_size)
        output_u = (all_memory_values*alpha_u).sum(dim=1)
        output_u = torch.cat([output_u, last_memory_values], dim=1)
        output_u = self.dropout(self.Explore_user(output_u))

        candidates_mask = get_candidate_mask(item_seq, self.device, self.n_items)
        output_u = output_u.masked_fill(candidates_mask.bool(), float('-inf'))
        explore_user_decoder = self.sigmoid(output_u)

        return explore_user_decoder

class Explore_Popular(nn.Module):
    def __init__(self, device, hidden_size, seq_len, num_item, dropout_prob):
        super(Explore_Popular, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_items = num_item
        self.device = device

        self.U_popular = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_popular = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.V_popular = nn.Linear(hidden_size, 1)

        self.Explore_popular = nn.Linear(hidden_size*2, num_item, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        all_memory_values, last_memory_values = all_memory, last_memory

        all_memory = self.dropout(self.U_popular(all_memory))
        last_memory = self.dropout(self.W_popular(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_ep = self.tanh(all_memory+last_memory)

        output_ep = self.V_popular(output_ep).squeeze(-1)
        if mask is not None:
            output_ep.masked_fill_(mask, -1e9)
        output_ep = output_ep.unsqueeze(-1)

        alpha_p = nn.Softmax(dim=1)(output_ep)
        alpha_p = alpha_p.repeat(1, 1, self.hidden_size)
        output_p = (all_memory_values*alpha_p).sum(dim=1)
        output_p = torch.cat([output_p, last_memory_values], dim=1)
        output_p = self.dropout(self.Explore_popular(output_p))

        candidates_mask = get_candidate_mask(item_seq, self.device, self.n_items)
        output_p = output_p.masked_fill(candidates_mask.bool(), float('-inf'))
        explore_popular_decoder = self.sigmoid(output_p)

        return explore_popular_decoder

def build_map(b_map, device, max_index = None):
    '''b_map: batch, seq_len   -> b_map_: batch, seq_len, item_num. A seq of one-hot.'''
    batch_size, b_len = b_map.size()
    if max_index is None:
        max_index = b_map.max() + 1
    if torch.cuda.is_available():
        b_map_ = torch.FloatTensor(batch_size, b_len, max_index).fill_(0).to(device)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max_index)
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    b_map_.requires_grad = False
    return b_map_

def get_candidate_mask(candidates, device, max_index=None):
    '''Candidates is the output of basic models or repeat or popular'''
    batch_size = len(candidates)
    if torch.cuda.is_available():
        candidates_mask = torch.FloatTensor(batch_size, max_index).fill_(1.0).to(device)
    else:
        candidates_mask = torch.ones(batch_size, max_index)
    for ind in range(batch_size):
        candidates_mask[ind].scatter_(0, torch.as_tensor(candidates[ind]).to(device), 0)
    candidates_mask.requires_grad = False
    return candidates_mask.bool()

def get_timeline_mask(batch_basket_emb, device, emb_size):
    batch_mask = []
    for basket_seq in batch_basket_emb:
        mask = []
        for basket_emb in basket_seq:
            if torch.equal(basket_emb, torch.zeros(emb_size).to(device)):
                mask.append(1)
            else:
                mask.append(0)
        batch_mask.append(torch.as_tensor(mask).bool())
    batch_mask = torch.stack(batch_mask, 0).to(device)
    return batch_mask.bool()

def get_label_tensor(labels, device, max_index=None):
    '''Candidates is the output of basic models or repeat or popular
    labels is list[]'''
    batch_size = len(labels)
    if torch.cuda.is_available():
        label_tensor = torch.FloatTensor(batch_size, max_index).fill_(0.0).to(device)
    else:
        label_tensor = torch.zeros(batch_size, max_index)
    for ind in range(batch_size):
        if len(labels[ind])!=0:
            label_tensor[ind].scatter_(0, torch.as_tensor(labels[ind]).to(device), 1)
    label_tensor.requires_grad = False # because this is not trainable
    return label_tensor

def get_sub_label_tensor(labels, candidates, device, max_index=None):
    batch_size = labels.size(0)
    if torch.cuda.is_available():
        label_tensor = torch.FloatTensor(batch_size, max_index).fill_(0.0).to(device)
    else:
        label_tensor = torch.zeros(batch_size, max_index)
    for ind in range(batch_size):
        # sub_labels = list(set(labels[ind])&set(candidates[ind]))
        sub_labels = torch.as_tensor([item for item in labels[ind] if item in candidates[ind]])
        label_tensor[ind].scatter_(0, sub_labels.to(device), 1.)
    label_tensor.requires_grad = False # because this is not trainable
    return label_tensor

def get_sub_label_set(labels, candidates):
    batch_size = len(labels)
    batch_label_set = []
    # print('Labels', labels[0])
    # print("Candiates:", candidates[0])
    for ind in range(batch_size):
        # sub_labels = list(set(labels[ind]&candidates[ind]))
        sub_labels = [item for item in labels[ind] if item in candidates[ind]]
        batch_label_set.append(sub_labels)
    # batch_sub_labels = torch.stack(batch_label_set, dim=0).to(device)
    # print('cand:', batch_label_set[0])
    return batch_label_set