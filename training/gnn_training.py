import numpy as np
import random
import torch
import torch.nn.functional as F
import pickle
import math
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import manifold
from tqdm import tqdm  
import matplotlib.pyplot as plt
from network_data.data_preprocessing import load_data
from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from models.fusion import Fusion
from models.encoder_temporal import Encoder
from models.encoder_hybrid import Encoder as Encoder_2
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge
from utils.utils import seed_torch
from utils.utils import UnsupervisedLoss
from utils.utils import distinguish_loss


#import sys
#np.set_printoptions(threshold=sys.maxsize)
#torch.set_printoptions(threshold=10_100000)

def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def genMetrics(output, labels):
    #preds = output.max(1)[1].type_as(labels)
    preds = np.argmax(output, axis=1)
    print("preds ", preds)
    print("labels", labels)
    recall = recall_score(labels, preds, average='macro')
    macroprec = precision_score(labels, preds, average='macro')
    macrof1 = f1_score(labels, preds, average='macro') 
    return recall, macroprec, macrof1


def attns_visual(attns, edge_num, edge_value):
    print("edge_num", edge_num)
    print("edge_value", edge_value)
    attns = attns.cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attns[150-edge_num:, 150-edge_num:], cmap='Blues') 
    fig.colorbar(cax)
    plt.show()
    plt.clf()
    

def train_gnn_y(args, device=torch.device('cpu')):
    print('################################data preprocessing##################################')
    seed_torch(args.seed)
    ##data loading
    data = load_data(args, 0)
    
    ## create tensor for transformer encoder  
    # for explanation
    edges_trans_kfold = []
    edges_timeDiff_kfold = []
    edges_trans_kfold_test = []
    edges_timeDiff_kfold_test = []
    timeDiff_count_kfold_test = []
    timeDiff_value_kfold_test = []
    for k in range(args.kfold):
        #train
        edges_transformer = []
        edges_timeDiff = []
        for i, mask in enumerate(data.train_y_mask[k]):
            if mask == True:
                # Compute the positional encodings once in log space.
                edge_trans = data.edge_trans_train_kfold[k][i]
                edge_len = len(edge_trans)
                dim_pos = 16
                pe = torch.zeros(edge_len, dim_pos)
                position = torch.arange(0, edge_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, dim_pos, 2) * - (math.log(10000.0) / dim_pos))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                time_diff = torch.tensor(edge_trans)
                max_len = 150
                if pe.size(0) >= max_len:
                    zero_pad_len = 0
                    pe = pe[pe.size(0)-max_len:,:]
                    tim_diff = time_diff[time_diff.size(0)-max_len:, :]
                else:
                    zero_pad_len = max_len - pe.size(0)
                    zero_pe = torch.zeros(zero_pad_len, dim_pos)
                    zero_time_diff = torch.zeros(zero_pad_len, 1)
                    pe = torch.cat([zero_pe, pe], dim=0)
                    tim_diff = torch.cat([zero_time_diff, time_diff], dim=0)
                edges_transformer.append(pe)
                edges_timeDiff.append(tim_diff)
        edges_trans_kfold.append(torch.stack(edges_transformer))
        edges_timeDiff_kfold.append(torch.stack(edges_timeDiff))
        
        #test
        edges_transformer = []
        edges_timeDiff = []
        timeDiff_count = []
        timeDiff_value = []
        for i, mask in enumerate(data.test_y_mask[k]):
            if mask == True:
                # Compute the positional encodings once in log space.
                edge_trans = data.edge_trans_test_kfold[k][i]
                edge_len = len(edge_trans)
                dim_pos = 16
                pe = torch.zeros(edge_len, dim_pos)
                position = torch.arange(0, edge_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, dim_pos, 2) * - (math.log(10000.0) / dim_pos))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                time_diff = torch.tensor(edge_trans)
                
                ## padding
                if pe.size(0) >= max_len:
                    zero_pad_len = 0
                    pe = pe[pe.size(0)-max_len:,:]
                    tim_diff = time_diff[time_diff.size(0)-max_len:, :]
                    timeDiff_count.append(max_len)
                    timeDiff_value.append(time_diff[time_diff.size(0)-max_len:, :].view(1,-1))
                else:
                    timeDiff_count.append(pe.size(0))
                    timeDiff_value.append(time_diff.view(1,-1))
                    zero_pad_len = max_len - pe.size(0)
                    zero_pe = torch.zeros(zero_pad_len, dim_pos)
                    zero_time_diff = torch.zeros(zero_pad_len, 1)
                    pe = torch.cat([zero_pe, pe], dim=0)
                    tim_diff = torch.cat([zero_time_diff, time_diff], dim=0)
                edges_transformer.append(pe)
                edges_timeDiff.append(tim_diff)
        edges_trans_kfold_test.append(torch.stack(edges_transformer))
        edges_timeDiff_kfold_test.append(torch.stack(edges_timeDiff))
        timeDiff_count_kfold_test.append(timeDiff_count)
        timeDiff_value_kfold_test.append(timeDiff_value)

    
    n_row = data.user_num
    n_col = data.product_num
    x = data.x.clone().detach().to(device)
    edge_index = data.edge_index.detach().to(device)
    edge_attr = data.edge_attr.detach().to(device)
    train_edge_index = data.train_edge_index
    train_edge_attr = data.train_edge_attr
    test_edge_index = data.test_edge_index
    test_edge_attr = data.test_edge_attr
    dict_prev = data.dict_user_pro

    print('################################training starts##################################')
    K_Fold = len(data.train_labels)
    print("K_Fold", K_Fold)
    num_item = 3
    AUC_max_list = []
    for k in range(K_Fold):
        print("K-th", k)
        AUC_list = []
        ## k-th train and test labels
        edge_index = data.train_edge_index.detach().to(device)
        edge_attr = data.train_edge_attr.detach().to(device)
        train_y_labels = data.train_labels[k].clone().detach().to(device)
        train_y_mask = data.train_y_mask[k].clone().detach().to(device)
        edges_transformer = edges_trans_kfold[k].detach().to(device)
        edges_timeDiff = edges_timeDiff_kfold[k].detach().to(device)
        edge_index_test = data.test_edge_index.detach().to(device)
        edge_attr_test = data.test_edge_attr.detach().to(device)
        test_y_labels = data.test_labels[k].clone().detach().to(device)
        test_y_mask = data.test_y_mask[k].clone().detach().to(device)
        edges_transformer_test = edges_trans_kfold_test[k].detach().to(device)
        edges_timeDiff_test = edges_timeDiff_kfold_test[k].detach().to(device)

        model_sageGGNN = get_gnn(args).to(device)
        model_fuse = Fusion().to(device)
        model_trans_encoder = Encoder(16, args.node_dim, args.node_dim, max_len).to(device)
        model_trans_encoder_2 = Encoder_2(args.edge_dim, args.node_dim).to(device)
        predict_model = MLPNet(args.node_dim, 2).to(device)

        trainable_parameters =  list(model_sageGGNN.parameters()) \
                              + list(model_trans_encoder.parameters()) \
                              + list(model_trans_encoder_2.parameters()) \
                              + list(model_fuse.parameters()) \
                              + list(predict_model.parameters())

        # build optimizer
        scheduler, opt = build_optimizer(args, trainable_parameters)

        # train
        Train_loss = []
        
        print("all y num is {}, train num is {}, test num is {}"\
                .format(
                train_y_mask.shape[0],torch.sum(train_y_mask),
                torch.sum(test_y_mask)))

        for epoch in tqdm(range(args.epochs)):

            model_sageGGNN.train()
            predict_model.train()
            opt.zero_grad()
            
            x_full_hist = []
            edge_attr_diff = edge_attr[:,4].unsqueeze(-1) ##for encoder_2
            x_embd, edge_attr_update = model_sageGGNN(x, edge_attr[:,:7], edge_index)    
            edge_attr_merge = torch.cat([edge_attr_diff, edge_attr_update], dim=-1) ##for encoder_2
           
            x_trans_encoder, _ = model_trans_encoder(edges_timeDiff)
            x_trans_encoder_2 = model_trans_encoder_2(x_embd, edge_attr_merge, edge_index)
            x_fuse = model_fuse(x_embd[train_y_mask], x_trans_encoder, x_trans_encoder_2[train_y_mask])
            pred = predict_model(x_fuse)
            pred_train = pred
            label_train = train_y_labels
            
            correct_pred_train = correct_prediction(pred_train, label_train)/len(label_train)

            ## computing loss
            loss = F.cross_entropy(pred_train, label_train.long())
            loss.backward()
            opt.step()
            train_loss = loss.item()
            if scheduler is not None:
                scheduler.step(epoch)
                
            '''
            #for AUC
            pred_train_np = pred_train.detach().numpy()
            label_train_np = label_train.detach().numpy()
            pred_train_select = [entry[label_train_np[idx_train]] for idx_train, entry in enumerate(pred_train_np)]
            pred_train_select = np.array(pred_train_select)
            '''

            predict_model.eval()
            with torch.no_grad():
                edge_attr_diff = edge_attr_test[:,4].unsqueeze(-1) ##for encoder_2
                x_embd, edge_attr_update = model_sageGGNN(x, edge_attr_test[:,:7], edge_index_test)
                edge_attr_merge = torch.cat([edge_attr_diff, edge_attr_update], dim=-1) ##for encoder_2
                x_trans_encoder, weights = model_trans_encoder(edges_timeDiff_test)
                x_trans_encoder_2 = model_trans_encoder_2(x_embd, edge_attr_merge, edge_index_test)
                x_fuse = model_fuse(x_embd[test_y_mask], x_trans_encoder, x_trans_encoder_2[test_y_mask])

                pred = predict_model(x_fuse)
                pred_test = pred
                label_test = test_y_labels

                #for AUC
                pred_test_np = pred_test.cpu().numpy()
                pred_test = F.softmax(pred_test, dim=-1)
                label_test_np = label_test.cpu().numpy()
                pred_test_select = [entry[1] for idx_test, entry in enumerate(pred_test_np)]
                pred_test_select = np.array(pred_test_select)

                #Accuracy
                correct_pred_test = correct_prediction(pred_test, label_test)/len(label_test)
                        
                #AUC
                AUC_test = roc_auc_score(label_test_np, pred_test_select)
                AUC_list.append(AUC_test)
                
        AUC_max_list.append(max(AUC_list))
        print("AUC", AUC_max_list)  
        print('#################################################################################')
    print("AVE AUC", np.mean(AUC_max_list))
