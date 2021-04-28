import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os.path as osp
import torch
import subprocess
import random
import os 


def np_random(seed=None):
    rng = np.random.RandomState()
    rng.seed(seed)
    return rng

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return scheduler, optimizer

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def save_mask(length,true_rate,log_dir,seed):
    np.random.seed(seed)
    mask = np.random.rand(length) < true_rate
    np.save(osp.join(log_dir,'len'+str(length)+'rate'+str(true_rate)+'seed'+str(seed)),mask)
    return mask

def get_known_mask(known_prob, edge_num):
    known_mask = (torch.FloatTensor(edge_num, 1).uniform_() < known_prob).view(-1)
    return known_mask

def mask_edge(edge_index,edge_attr,mask,remove_edge):
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:,mask] 
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.
    return edge_index, edge_attr

def one_hot(batch,depth):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,torch.tensor(batch,dtype=int))

def soft_one_hot(batch,depth):
    batch = torch.tensor(batch)
    encodings = torch.zeros((batch.shape[0],depth))
    for i,x in enumerate(batch):
        for r in range(depth):
            encodings[i,r] = torch.exp(-((x-float(r))/float(depth))**2)
        encodings[i,:] = encodings[i,:]/torch.sum(encodings[i,:])
    return encodings

def construct_missing_X_from_mask(train_mask, df):
    nrow, ncol = df.shape
    data_incomplete = np.zeros((nrow, ncol))
    data_complete = np.zeros((nrow, ncol)) 
    train_mask = train_mask.reshape(nrow, ncol)
    for i in range(nrow):
        for j in range(ncol):
            data_complete[i,j] = df.iloc[i,j]
            if train_mask[i,j]:
                data_incomplete[i,j] = df.iloc[i,j]
            else:
                data_incomplete[i,j] = np.NaN
    return data_complete, data_incomplete

def construct_missing_X_from_edge_index(train_edge_index, df):
    nrow, ncol = df.shape
    data_incomplete = np.zeros((nrow, ncol))
    data_complete = np.zeros((nrow, ncol)) 
    train_edge_list = torch.transpose(train_edge_index,1,0).numpy()
    train_edge_list = list(map(tuple,[*train_edge_list]))
    for i in range(nrow):
        for j in range(ncol):
            data_complete[i,j] = df.iloc[i,j]
            if (i,j) in train_edge_list:
                data_incomplete[i,j] = df.iloc[i,j]
            else:
                data_incomplete[i,j] = np.NaN
    return data_complete, data_incomplete

# get gpu usage
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def auto_select_gpu(memory_threshold = 7000, smooth_ratio=200, strategy='greedy'):
    gpu_memory_raw = get_gpu_memory_map() + 10
    if strategy=='random':
        gpu_memory = gpu_memory_raw/smooth_ratio
        gpu_memory = gpu_memory.sum() / (gpu_memory+10)
        gpu_memory[gpu_memory_raw>memory_threshold] = 0
        gpu_prob = gpu_memory / gpu_memory.sum()
        cuda = str(np.random.choice(len(gpu_prob), p=gpu_prob))
        print('GPU select prob: {}, Select GPU {}'.format(gpu_prob, cuda))
    elif strategy == 'greedy':
        cuda = np.argmin(gpu_memory_raw)
        print('GPU mem: {}, Select GPU {}'.format(gpu_memory_raw[cuda], cuda))
    return cuda


def get_hete_adg_mat(edge_index, edge_attr, dict_user_pro, mat_type, neigh_num):
    ## record node with edges index
    node_edge_dict = {}
    for i, node_idx in enumerate(edge_index.permute(1,0)):
        if node_idx[1].item() not in node_edge_dict.keys():
            node_edge_dict[node_idx[1].item()] = [(node_idx[0].item(), edge_attr[i])]
        else:
            node_edge_dict[node_idx[1].item()].append((node_idx[0].item(), edge_attr[i]))
    ##adj matrix for burstiness
    if mat_type == 'burst':
        for node_idx in node_edge_dict:
            node_edge_dict[node_idx] = sorted(node_edge_dict[node_idx], key=lambda x: x[1][4].item())
    elif mat_type == 'current':
        for node_idx in node_edge_dict:
            node_edge_dict[node_idx].reverse()
    if mat_type == 'calm':
        for node_idx in node_edge_dict:
            node_edge_dict[node_idx] = sorted(node_edge_dict[node_idx], key=lambda x: x[1][4].item(), reverse=True)
            
    edge_index_burst = []
    edge_attr_burst = []

    for user in dict_user_pro:
        if 'u' in user:
          x_i = dict_user_pro[user]
          for j, x_j in enumerate(node_edge_dict[x_i]):
              if j == neigh_num:
                  break
              ## product->user
              edge_index_burst.append([x_j[0], x_i])
              edge_attr_burst.append(x_j[1])
              ## user->product
              edge_index_burst.append([x_i, x_j[0]])
              edge_attr_burst.append(x_j[1])
              '''
              for pro_attr in node_edge_dict[x_j[0]]:
                  if pro_attr[0] == x_i:
                      edge_attr_burst.append(pro_attr[1])
              '''
    edge_index_burst = torch.tensor(edge_index_burst).permute(1,0)
    edge_attr_burst = torch.stack(edge_attr_burst)
    return edge_index_burst, edge_attr_burst

def generate_uupp_graph(edge_index, user_prod_inv_dict):
    ##generated user-user & product-product graph
    uupp_dict = {}
    for i, uupp_edge in enumerate(edge_index.permute(1,0)):
        if uupp_edge[0].item() not in uupp_dict.keys():
            uupp_dict[uupp_edge[0].item()] = [uupp_edge[1].item()]
        else:
            uupp_dict[uupp_edge[0].item()].append(uupp_edge[1].item())
    print("uupp_dict", len(uupp_dict))
    edge_uu_idx = []
    edge_pp_idx = []
    ##user-user graph
    for uupp_key in uupp_dict:
        ##user-user graph
        if 'u' in user_prod_inv_dict[uupp_key]:
          user_unique = []
          for product_idx in uupp_dict[uupp_key]:
              ##list all user for same product
              user4pro_all = uupp_dict[product_idx]
              for user4pro in user4pro_all: 
                  if user4pro not in user_unique:
                      user_unique.append(user4pro)
                      uu = []
                      uu.append(uupp_key)
                      uu.append(user4pro)
                      edge_uu_idx.append(uu)
        ##product-product graph
        else:
          product_unique = []
          for user_idx in uupp_dict[uupp_key]:
              ##list all product for same user
              pro4user_all = uupp_dict[user_idx]
              for pro4user in  pro4user_all:
                  if pro4user not in product_unique:
                      product_unique.append(pro4user)
                      pp = []
                      pp.append(uupp_key)
                      pp.append(pro4user)
                      edge_pp_idx.append(pp)
    edge_uu_idx = torch.tensor(edge_uu_idx).permute(1,0)
    edge_pp_idx = torch.tensor(edge_pp_idx).permute(1,0)

    return (edge_uu_idx, edge_pp_idx)

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class UnsupervisedLoss(object):
    def __init__(self, edge_uu_idx_burst, edge_uu_idx_curr, sample_num, Q):
        super(UnsupervisedLoss, self).__init__()
        self.uu_dict = {}
        self.sample_num = sample_num
        self.Q = Q
        for uu in edge_uu_idx_burst.permute(1,0):
            if uu[0].item() not in self.uu_dict.keys():
                self.uu_dict[uu[0].item()] = [uu[1].item()]
            else:
                self.uu_dict[uu[0].item()].append(uu[1].item())
        for uu in edge_uu_idx_curr.permute(1,0):
            if uu[0].item() not in self.uu_dict.keys():
                self.uu_dict[uu[0].item()] = [uu[1].item()]
            else:
                self.uu_dict[uu[0].item()].append(uu[1].item())
        for key in self.uu_dict:
            self.uu_dict[key] = np.unique(self.uu_dict[key])
        print("self.uu_dict", len(self.uu_dict))
        
    def get_loss(self, x, sample_num, uu_dict):
        sample_pair_p = {}
        sample_pair_n = {}

        ## get positive sample pairs
        for key in uu_dict:
            if len(uu_dict[key]) >= sample_num:
                rand_p = np.random.randint(len(uu_dict[key]), size=(sample_num))
            else:
                rand_p = np.array(range(len(uu_dict[key])))
            sample_pair_p[key] = uu_dict[key][rand_p]
      
        ## get negative sample pairs
        for key in uu_dict:
            nodes_neg = []
            for node_neg in uu_dict.keys():
                if node_neg not in uu_dict[key]:
                    nodes_neg.append(node_neg)
            rand_p = np.random.randint(len(nodes_neg), size=(sample_num))
            sample_pair_n[key] = np.array(nodes_neg)[rand_p]
        
        score_p_dict = {}
        score_n_dict = {}
        score_node = []
        for pair1 in sample_pair_p:
            ##computing positvie score
            for pair2 in sample_pair_p[pair1]:
                score_p = F.cosine_similarity(x[pair1].unsqueeze(0), x[pair2].unsqueeze(0))
                score_p = torch.log(torch.sigmoid(score_p))
                if pair1 not in score_p_dict.keys():
                    score_p_dict[pair1] = [score_p]
                else:
                    score_p_dict[pair1].append(score_p)
            score_p_dict[pair1] = torch.stack(score_p_dict[pair1])
            ##computing negative score
            for pair2 in sample_pair_n[pair1]:
                score_n = F.cosine_similarity(x[pair1].unsqueeze(0), x[pair2].unsqueeze(0))
                score_n = torch.log(torch.sigmoid(-score_n))
                if pair1 not in score_n_dict.keys():
                    score_n_dict[pair1] = [score_n]
                else:
                    score_n_dict[pair1].append(score_n)
            score_n_dict[pair1] = torch.mean(torch.stack(score_n_dict[pair1]))
        ## computing loss
        for key in score_p_dict:
            score_node.append(torch.mean(-score_p_dict[key] - score_n_dict[key]))
        loss = torch.mean(torch.stack(score_node))
        return loss

class distinguish_loss(object):
    def __init__(self, dict_user_pro):
        super(distinguish_loss, self).__init__()
        self.uu_mask = []

        for key in dict_user_pro:
            if 'u' in key:
                self.uu_mask.append(True)
            else:
                self.uu_mask.append(False)
        self.uu_mask = torch.tensor(self.uu_mask)

    def get_loss(self, x_embd_burst, x_embd_curr):
        #x_uu_burst = x_embd_burst[self.uu_mask]
        #x_uu_curr = x_embd_curr[self.uu_mask]
        x_uu_burst = x_embd_burst
        x_uu_curr = x_embd_curr
        score = F.cosine_similarity(x_uu_burst,  x_uu_curr)
        score = torch.log(torch.sigmoid(-score))
        loss = torch.mean(score)
        return -loss
