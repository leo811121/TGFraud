import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from tqdm import tqdm

import torch
import random
import numpy as np
import pdb
import pickle
import statistics as stat
import math
import datetime
import os
 
from utils.utils import get_known_mask, mask_edge


## ave, gap of timestamp for user
def creat_user_timestamp(edges, user_prod_dict, UorP=True):
    if UorP:
        item = 0
    else:
        item = 1

    user_min_dict = {}
    for idx, edge in enumerate(edges):
        user_index = user_prod_dict[edge[item]]
        if user_index not in user_min_dict: 
            user_min_dict[user_index] = {}
            user_min_dict[user_index]['user_iden'] = edge[item]
            user_min_dict[user_index]['timegap'] = []
            user_min_dict[user_index]['timestamp'] = [edge[2]['timestamp']]
        elif user_index in user_min_dict:
            user_min_dict[user_index]['user_iden'] = edge[item]
            user_min_dict[user_index]['timestamp'].append(edge[2]['timestamp'])
    for key in user_min_dict:
        user_min_dict[key]['timestamp'] = np.asarray(sorted(user_min_dict[key]['timestamp']))
        timestamp_prev = user_min_dict[key]['timestamp'][0]
        for timestamp in user_min_dict[key]['timestamp']:
            user_min_dict[key]['timegap'].append(timestamp - timestamp_prev)
            timestamp_prev = timestamp
        user_min_dict[key]['timegap'] = np.asarray(user_min_dict[key]['timegap'])
    for key in user_min_dict:
        timestamp_ave = stat.mean(user_min_dict[key]['timestamp'])
        timegap_ave = stat.mean(user_min_dict[key]['timegap'])
        user_gap = max(user_min_dict[key]['timestamp']) - min(user_min_dict[key]['timestamp'])
        user_min_dict[key]['stamp_ave'] = timestamp_ave
        user_min_dict[key]['stamp_ave_day'] = timestamp_ave/(3600*24)
        user_min_dict[key]['gap_ave'] = timegap_ave
        user_min_dict[key]['gap_ave_day'] = timegap_ave/(3600*24)
        user_min_dict[key]['gap_max'] = user_gap
        user_min_dict[key]['gap_max_day'] = user_gap/(3600*24)
    for key in user_min_dict:
        user_min_dict[key]['timegap_entro'] = []
        user_min_dict[key]['timegap_day'] = []
        for timegap in user_min_dict[key]['timegap']:
            if timegap!=0. and user_min_dict[key]['gap_max']!=0:
                user_min_dict[key]['timegap_entro'].append((timegap/user_min_dict[key]['gap_max']) * math.log(timegap/user_min_dict[key]['gap_max']))
            else:
                user_min_dict[key]['timegap_entro'].append(0.)
            user_min_dict[key]['timegap_day'].append(timegap/(3600*24))
        user_min_dict[key]['timegap_entro_all'] = -sum(user_min_dict[key]['timegap_entro'])
    return user_min_dict

def create_node(user_prod_dict, edge_index, edge_attr): 
    node_init = np.zeros((len(user_prod_dict), 23))
    rating_list = {}
    for i in range(len(user_prod_dict)):
        rating_list[i] = []
    #rating cumulate
    for idx, edge in enumerate(edge_attr):
        rate = edge[7].item()
        rate = math.floor(rate*10)/10
        node_init[edge_index[0][idx].item()][int(rate*10)+10] +=1
        #product nums
        node_init[edge_index[0][idx].item()][21] +=1
        rating_list[edge_index[0][idx].item()].append(rate)
    #rating entropy
    for idx, node in enumerate(node_init):
        entropy = 0
        rate_sum = sum(node[:21])
        for rate_num in range(21):
            if node[rate_num] == 0:
                entropy += 0.
            else:
                p = node[rate_num]/rate_sum
                entropy -= p*math.log(p)
        node_init[idx][22] = entropy
        #median mean max
        #node_init[idx][23] = stat.median(rating_list[idx])
        #node_init[idx][24] = stat.mean(rating_list[idx])
        #node_init[idx][25] = max(rating_list[idx])
    node_init = torch.tensor(node_init, dtype=torch.float)

    ##mask rating related features
    node_rating_mask = [False for i in range(23)]
    node_rating_mask[21] = True
    node_rating_mask = torch.tensor(node_rating_mask).repeat(node_init.size()[0],1)
    node_init = node_init[node_rating_mask].view(-1,1)
    return node_init
  

def get_data(args, nodes, edges, badusers, goodusers, seed=0):
    ## generate user & product graph
    user_names = [node for node in nodes if "u" in node]
    product_names = [node for node in nodes if "p" in node]
    #all_names = user_names + product_names
    user_prod_dict = {}
    user_prod_inv_dict = {}
    for idx_up, node_up in enumerate(nodes):
        user_prod_dict[node_up] = idx_up
        user_prod_inv_dict[idx_up] = node_up

    ## ave, gap of timestamp for user
    user_min_dict = creat_user_timestamp(edges, user_prod_dict, UorP=True)
    ## ave, gap of timestamp for product
    product_min_dict = creat_user_timestamp(edges, user_prod_dict, UorP=False)

    ## generate edge vector
    start = []
    end = []
    edge_attr = []
    ## sort edge by timestamp
    edges = sorted(edges, key=lambda x: x[2]['timestamp'])
    user_edge_count = {}
    for u_or_p in user_prod_dict:
        user_edge_count[user_prod_dict[u_or_p]] = 0
    for edge_idx, edge in enumerate(edges):
        if edge[0] in user_prod_dict and edge[1] in user_prod_dict:
            start.append(user_prod_dict[edge[0]])
            end.append(user_prod_dict[edge[1]])
            ### year month date hr
            time_info = datetime.datetime.fromtimestamp(edge[2]['timestamp']).isoformat()
            time_info = time_info.split('-')
            time_info[0] = int(time_info[0][2:])
            time_info[1] = int(time_info[1])
            date = time_info[2].split('T')[0]
            hr = time_info[2].split('T')[1].split(':')[0]
            time_info[2] = int(date)
            time_info.append(int(hr))
            ###timestamp of user and product
            index_time_user = user_prod_dict[edge[0]]
            index_time_prod = user_prod_dict[edge[1]]
            timegap_user = user_min_dict[index_time_user]['timegap_day'][user_edge_count[index_time_user]]
            timegap_product = product_min_dict[index_time_prod]['timegap_day'][user_edge_count[index_time_prod]]
            user_edge_count[index_time_user] += 1
            user_edge_count[index_time_prod] += 1
            time_info.append(timegap_user)
            time_info.append(timegap_product)
            ###original timstamp
            time_info.append(edge[2]['timestamp']/math.pow(10, 8))
            ###rating
            time_info.append(edge[2]['weight'])                
            edge_attr.append(time_info)
    star_end = start + end
    end_start = end + start
    edge_index = torch.tensor([star_end, end_start], dtype=int)
    edge_attr = torch.tensor(edge_attr + edge_attr, dtype=torch.float)
    print("edge_index", edge_index.size())
    print("edge_attr", edge_attr.size())
    
    ## for inductive learning
    inductive_len = int(edge_attr.size(0)*args.inductive_rate)
    edge_index_ind = edge_index[:, :inductive_len]
    edge_attr_ind = edge_attr[:inductive_len, :]
    x_ind = set()
    for idx_ind in edge_index_ind.permute(1,0):
        x_ind.add(idx_ind[0].item())
        x_ind.add(idx_ind[1].item())
    print("edge_index_ind", edge_index_ind.size())
    print("edge_attr_ind", edge_attr_ind.size())

    ##node initializing
    print("node initializing")
    x = create_node(user_prod_dict, edge_index, edge_attr)
    print("x", x.size())


    ## mask rating 
    mask_dir_list = []
    for i, edg_idx in enumerate(edge_index.permute(1,0)):
        mask_dir = [True]*edge_attr.size()[1]
        #mask rating
        mask_dir[-1] = False
        mask_dir_list.append(mask_dir)
    mask_dir_list = torch.tensor(mask_dir_list)
    edge_attr = torch.masked_select(edge_attr, mask_dir_list).view(-1,7)
    print("edge_attr w/o rating", edge_attr.size())
    
    ## add node info to edge
    edge_attr_node = []
    for i, edg_idx in enumerate(edge_index.permute(1,0)):
        edge_attr_node.append(torch.cat([edge_attr[i], x[edg_idx[1]]], dim=-1))
    edge_attr = torch.stack(edge_attr_node)


    ## generate label
    users_bad = []
    users_bad_key = []
    users_good = []
    users_good_key = []
    for key in user_prod_dict:
        if key in badusers:
            users_bad.append(1)
            users_bad_key.append(user_prod_dict[key])
        elif key in goodusers:
            users_good.append(0)
            users_good_key.append(user_prod_dict[key])

    user_all = users_good + users_bad
    user_key_all = users_good_key +  users_bad_key
    
    sfolder = StratifiedKFold(n_splits=10,random_state=3,shuffle=True)
    train_labels = []
    train_y_mask = []
    test_labels = []
    test_y_mask = []
    tsne_label = []
    tsne_mask = []
    for train, test in sfolder.split(user_all, user_all):
        ##train
        train_labels_s = [2]*len(user_prod_dict)
        train_y_mask_s = [False]*len(user_prod_dict)
        for user_label in train:
            train_labels_s[user_key_all[user_label]] = user_all[user_label]
            if user_label != 2:
                train_y_mask_s[user_key_all[user_label]] = True
            else:
                train_y_mask_s[user_key_all[user_label]] = False
        train_y_mask_s = torch.tensor(train_y_mask_s)
        train_labels_s = torch.tensor(train_labels_s)
        train_labels_s = train_labels_s[train_y_mask_s]
        train_labels.append(train_labels_s)
        train_y_mask.append(train_y_mask_s)

        ##test
        test_labels_s = [2]*len(user_prod_dict)
        test_y_mask_s = [False]*len(user_prod_dict)
        for user_label in test:
            test_labels_s[user_key_all[user_label]] = user_all[user_label]
            if user_label != 2:
                test_y_mask_s[user_key_all[user_label]] = True
            else:
                test_y_mask_s[user_key_all[user_label]] = False
        test_y_mask_s = torch.tensor(test_y_mask_s)
        test_labels_s = torch.tensor(test_labels_s)
        test_labels_s = test_labels_s[test_y_mask_s]
        test_labels.append(test_labels_s)
        test_y_mask.append(test_y_mask_s)
        
        #tsne
        tsne_labels_s = [2]*len(user_prod_dict)
        tsne_mask_s = [False]*len(user_prod_dict)
        tsne_list = np.concatenate((train, test), axis=-1)
        for i, user_label in enumerate(tsne_list):
            #print("tsne i", i)
            tsne_labels_s[user_key_all[user_label]] = user_all[user_label]
            if user_label != 2:
                tsne_mask_s[user_key_all[user_label]] = True
            else:
                tsne_mask_s[user_key_all[user_label]] = False
        tsne_label.append(torch.tensor(tsne_labels_s))
        tsne_mask.append(torch.tensor(tsne_mask_s))
    
    ## generate dataset_train for transformer_2
    print("preparing for temporal encoder")
    edge_trans_train_kfold = []
    for y_mask in tqdm(train_y_mask):
        edge_transformer_train = {}
        for mask_id, mask in enumerate(y_mask):
            if mask.item() == True:
                edge_transformer_train[mask_id] = []
        for i, edge in enumerate(edge_index_ind.permute(1,0)):
            if edge[0].item() in edge_transformer_train.keys():
                #(time difference, position)
                if edge_attr_ind[i][4].item() == 0.:
                    edge_transformer_train[edge[0].item()].append([10**-10])
                else:
                    edge_transformer_train[edge[0].item()].append([edge_attr_ind[i][4].item()])
        edge_trans_train_kfold.append(edge_transformer_train)
        
    ## generate dataset_test for transformer_2
    edge_trans_test_kfold = []
    for y_mask in tqdm(test_y_mask):
        edge_transformer_test = {}
        for mask_id, mask in enumerate(y_mask):
            if mask.item() == True:
                edge_transformer_test[mask_id] = []
        for i, edge in enumerate(edge_index.permute(1,0)):
            if edge[0].item() in edge_transformer_test.keys():
                #(time difference, position)
                if edge_attr[i][4].item() == 0.:
                    edge_transformer_test[edge[0].item()].append([10**-10])
                else:
                    edge_transformer_test[edge[0].item()].append([edge_attr[i][4].item()])
        edge_trans_test_kfold.append(edge_transformer_test)
        
    #set seed to fix known/unknwon edges
    torch.manual_seed(seed)
    
    ##generate train & test edge
    train_edge_index = edge_index_ind
    train_edge_attr = edge_attr_ind
    test_edge_index = edge_index
    test_edge_attr = edge_attr
    print("train_edge_index", train_edge_index.size())
    print("train_edge_attr", train_edge_attr.size())
    print("test_edge_index", test_edge_index.size())
    print("test_edge_attr", test_edge_attr.size())
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                train_y_mask=train_y_mask,           test_y_mask=test_y_mask, tsne_mask=tsne_mask,
                train_edge_index=edge_index_ind,   test_edge_index=test_edge_index,
                train_edge_attr=edge_attr_ind,     test_edge_attr=test_edge_attr,
                train_labels=train_labels,           test_labels=test_labels, tsne_label=tsne_label,
                edge_attr_dim=train_edge_attr.shape[-1],
                user_num=len(user_names),
                product_num=len(product_names),
                dict_user_pro = user_prod_dict,
                dict_user_pro_inv = user_prod_inv_dict,
                edge_trans_train_kfold = edge_trans_train_kfold,
                edge_trans_test_kfold = edge_trans_test_kfold
               )
    return data

def load_data(args, epoch):
    folder = args.dataset
    f_path = os.path.join('./network_data', folder, folder+'_gt.csv')
    f = open(f_path,"r")
    badusers = []
    goodusers = []
    for l in f:
        l = l.strip().split(",")

        if l[1] == "-1":
                badusers.append('u'+l[0])
        else:
                goodusers.append('u'+l[0])
    f.close()
    print("badusers", len(badusers))
    print("goodusers", len(goodusers))

    G_path = os.path.join('./network_data', folder, folder + '_network.pkl')
    with open(G_path, 'rb') as f:
      G = pickle.load(f)
    nodes = G.nodes()
    edges = G.edges(data=True)

    data = get_data(args, nodes, edges, badusers, goodusers, args.seed)
    return data