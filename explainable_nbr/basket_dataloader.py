
from torch.utils.data import Dataset, DataLoader
import json
from candidates import *

class BasketDataset(Dataset):

    def __init__(self, config, mode='train'):
        # load data.
        train_file = config['train_file']
        tgt_file = config['tgt_file']
        data_config_file = config['data_config_file']
        item_cand_file = config['item_candidate']
        user_cand_file = config['user_candidate']
        candidate_num = config['candidate_num']
        with open(train_file, 'r') as f:
            data_all = json.load(f)
        with open(tgt_file, 'r') as f:
            tgt_all = json.load(f)
        with open(data_config_file, 'r') as f:
            data_config = json.load(f)
        with open(item_cand_file, 'r') as f:
            item_cand_all = json.load(f)
        with open(user_cand_file, 'r') as f:
            user_cand_all = json.load(f)

        self.total_num = data_config['item_num'] #item total num
        self.cand_num = candidate_num
        self.mode = mode
        # Could do some filtering here
        self.data = []
        self.tgt = []
        self.item_candidates = []
        self.user_candidates = []
        for user in data_config[mode]:
            if len(data_all[user])<=config['max_len']+2: #max_len set to 50
                self.data.append(data_all[user][1:-1])
                self.tgt.append(tgt_all[user][1])
                self.item_candidates.append(item_cand_all[user][:candidate_num])
                self.user_candidates.append(user_cand_all[user][:candidate_num])
        self.user_sum = len(self.data)
        self.data_pop = []
        for user in data_config['train']:#since we only have the
            if len(data_all[user])<=config['max_len']+2: #max_len set to 50
                self.data_pop.append(data_all[user][1:-1])
        self.repeat_candidates = get_repeat_candidates(self.data)
        self.explore_candidates = get_explore_candidates(self.data, self.total_num)
        self.popular_candidates = get_popular_candidates(self.data_pop, self.user_sum, self.total_num, self.cand_num)

    def __getitem__(self, ind):
        return self.data[ind], self.tgt[ind], \
               self.repeat_candidates[ind], self.explore_candidates[ind], self.item_candidates[ind], \
               self.user_candidates[ind], self.popular_candidates

    def get_batch_data(self, s, e):
        return self.data[s:e], self.tgt[s:e], \
               self.repeat_candidates[s:e], self.explore_candidates[s:e], self.item_candidates[s:e], \
               self.user_candidates[s:e], self.popular_candidates[s:e]

    def __len__(self):
        return self.user_sum

def load_datafile(train_file, tgt_file, data_config_file):

    # load data.
    with open(train_file, 'r') as f:
        data = json.load(f)
    with open(tgt_file, 'r') as f:
        tgt = json.load(f)
    with open(data_config_file, 'r') as f:
        data_config = json.load(f)

    # Could do some filtering here
    data_train = []
    tgt_train = []
    for user in data_config['train']:
        data_train.append(data[user])
        tgt_train.append(tgt[user])
    # Valid
    data_val = []
    tgt_val = []
    for user in data_config['meta']:
        data_val.append(data[user])
        tgt_val.append(tgt[user])
    #test
    data_test = []
    tgt_test = []
    for user in data_config['test']:
        data_test.append(data[user])
        tgt_test.append(tgt[user])

    return data_train, tgt_train, data_val, tgt_val, data_test, tgt_test

if __name__ =='__main__':
    train_file = '../jsondata/Instacart_baskets_train.json'
    tgt_file = '../jsondata/Instacart_baskets_test.json'
    config_file = '../jsondata/Instacart_keyset.json'
    dataset_s = BasketDataset(train_file, tgt_file, config_file)
    print(dataset_s.get_batch_data(0,4))