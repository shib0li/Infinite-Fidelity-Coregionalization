import numpy as np
import copy
import random

import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from data.pde_solvers import *
from data.rawdata import *

from infras.randutils import *
from infras.misc import *

import pickle
from tqdm.auto import tqdm, trange
from sklearn.model_selection import KFold

# dataset = InfFidRawData(
#     domain='Heat', 
#     fid_min=8, 
#     fid_max=128, 
#     preload=os.path.join('data', '__raw__'),
# ) \

dict_domains = {
   
    'Poisson' : {
        'fid_min':8,
        'fid_max':128,
        'interp':'linear',
    },
    
    'Heat' : {
        'fid_min':8,
        'fid_max':128,
        'interp':'linear',
    },
    
    'Burgers' : {
        'fid_min':16,
        'fid_max':128,
        'interp':'cubic',
    },
}

class MFData2D:
    def __init__(self,
                 domain,
                 fid_min,
                 fid_max,
                 t_min,
                 t_max,
                 fid_list_tr=None,
                 fid_list_te=None,
                 ns_list_tr=None,
                 ns_list_te=None,
                ):
        
        self._init_mappings(t_min, t_max, fid_min, fid_max)
        
        self.interp = dict_domains[domain]['interp']
        
        self.raw_dataset = InfFidRawData(
            domain=domain, 
            fid_min=dict_domains[domain]['fid_min'], 
            fid_max=dict_domains[domain]['fid_max'],
            preload=os.path.join('data', '__raw__'),
        ) 
        
        self.input_dim = self.raw_dataset.input_dim
        
        self.fid_list_tr = copy.deepcopy(fid_list_tr)
        self.fid_list_te = copy.deepcopy(fid_list_te)

        self.ns_list_tr = copy.deepcopy(ns_list_tr)
        self.ns_list_te = copy.deepcopy(ns_list_te)
        
        self.dict_fid_to_ns_tr = {}
        self.dict_fid_to_ns_te = {}
        for fid, ns in zip(self.fid_list_tr, self.ns_list_tr):
            self.dict_fid_to_ns_tr[fid] = ns
        for fid, ns in zip(self.fid_list_te, self.ns_list_te):
            self.dict_fid_to_ns_te[fid] = ns

        self.t_list_tr = [self.func_fid_to_t(fid) for fid in self.fid_list_tr]
        self.t_list_te = [self.func_fid_to_t(fid) for fid in self.fid_list_te]

        #print(self.t_list_tr)
        #print(self.t_list_te)

        assert len(self.fid_list_tr) == len(self.ns_list_tr)
        assert len(self.fid_list_te) == len(self.ns_list_te)
        
    def _init_mappings(self, t_min, t_max, fid_min, fid_max):
        
        self.fid_min = fid_min
        self.fid_max = fid_max
        self.t_min = t_min
        self.t_max = t_max
        
        self.func_fid_to_t = lambda fid: \
            (fid-fid_min)*(t_max-t_min)/(fid_max-fid_min)
        
        self.func_t_to_fid = lambda t: \
            round((t-t_min)*(fid_max-fid_min)/(t_max-t_min) + fid_min)
        
        self.func_t_to_idx = lambda t: \
            round((t-t_min)*(fid_max-fid_min)/(t_max-t_min))
        
        fid_list_all = [fid for fid in range(self.fid_min, self.fid_max+1)]
        #cprint('r', self.fid_list)
        
        # sanity check 
        t_steps = 100
        t_span = np.linspace(t_min, t_max, t_steps)
        for i in range(t_span.size):
            t = t_span[i]
            fid = self.func_t_to_fid(t)
            idx = self.func_t_to_idx(t)
            t_rev = self.func_fid_to_t(fid)
            #cprint('r', '{:3f}-{}-{}'.format(t, fid, fid_list_all[idx]))
            err_t = np.abs(t-t_rev)
            #cprint('b', '{:.5f}-{:.5f}-{:.5f}'.format(t, t_rev, err_t))
            if fid != fid_list_all[idx]:
                raise Exception('Check the mappings of fids')
            #
            if err_t >= (t_max-t_min)/(fid_max-fid_min):
                raise Exception('Check the mappings of t')
            #
        #
        
        
#     def _extract_subset_by_fold(self, train=False, fold=0):
        
#         sub_dict_fid_to_X = {}
#         sub_dict_fid_to_y = {}
        
#         if train:
            
#             idx_all = np.arange(self.raw_dataset.ns_list_tr[0])
#             fold_subset_idx = generate_random_choice(idx_all, self.ns_list_tr[0], seed=fold)
            
#             for fid, ns in zip(self.fid_list_tr, self.ns_list_tr):
                
#                 nested_idx = np.sort(generate_random_choice(fold_subset_idx, ns, seed=100+fold**2+fid))
                
#                 Xs = self.raw_dataset.dict_fid_to_Xtr[fid][nested_idx,:]
#                 ys = self.raw_dataset.dict_fid_to_ytr[fid][nested_idx,:]
                
#                 #print(Xs.shape)
#                 #print(ys.shape)
                
#                 sub_dict_fid_to_X[fid] = Xs
#                 sub_dict_fid_to_y[fid] = ys
                
#                 fold_subset_idx = nested_idx
#         else:
            
#             idx_all = np.arange(self.raw_dataset.ns_list_te[0])
#             fold_subset_idx = generate_random_choice(idx_all, self.ns_list_te[0], seed=fold)
            
#             for fid, ns in zip(self.fid_list_te, self.ns_list_te):
#                 Xs = self.raw_dataset.dict_fid_to_Xte[fid][fold_subset_idx,:]
#                 ys = self.raw_dataset.dict_fid_to_yte[fid][fold_subset_idx,:]
                
#                 sub_dict_fid_to_X[fid] = Xs
#                 sub_dict_fid_to_y[fid] = ys
                
#                 #print(Xs.shape)
#                 #print(ys.shape)
#             #
#         #
        
#         return sub_dict_fid_to_X, sub_dict_fid_to_y

    def _extract_subset_by_fold(self, train, fold):
        
        sub_dict_fid_to_X = {}
        sub_dict_fid_to_y = {}
        
        if train:
            
            idx_all = np.arange(self.raw_dataset.ns_list_tr[0])
            fold_subset_idx = generate_random_choice(idx_all, self.ns_list_tr[0], seed=fold)
            
            for fid, ns in zip(self.fid_list_tr, self.ns_list_tr):
                
                #nested_idx = np.sort(generate_random_choice(fold_subset_idx, ns, seed=100+fold**2+fid))
                nested_idx = fold_subset_idx[:ns]
                
                Xs = self.raw_dataset.dict_fid_to_Xtr[fid][nested_idx,:]
                ys = self.raw_dataset.dict_fid_to_ytr[fid][nested_idx,:]
                
                #print(Xs.shape)
                #print(ys.shape)
                
                sub_dict_fid_to_X[fid] = Xs
                sub_dict_fid_to_y[fid] = ys
                
                fold_subset_idx = nested_idx
        else:
            
            idx_all = np.arange(self.raw_dataset.ns_list_te[0])
            fold_subset_idx = generate_random_choice(idx_all, self.ns_list_te[0], seed=fold)
            
            for fid, ns in zip(self.fid_list_te, self.ns_list_te):
                Xs = self.raw_dataset.dict_fid_to_Xte[fid][fold_subset_idx,:]
                ys = self.raw_dataset.dict_fid_to_yte[fid][fold_subset_idx,:]
                
                sub_dict_fid_to_X[fid] = Xs
                sub_dict_fid_to_y[fid] = ys
                
                #print(Xs.shape)
                #print(ys.shape)
            #
        #
        
        return sub_dict_fid_to_X, sub_dict_fid_to_y
    
    
    def get_data(self, fold, train=True, scale=True, tensor=True, device=torch.device('cpu')):
        
        X_list = {}
        y_list = {}
        
        dict_Xs, dict_ys = self._extract_subset_by_fold(train, fold)
        if train:
            fids = self.fid_list_tr
            dict_ns = self.dict_fid_to_ns_tr
            copy_t_list = copy.deepcopy(self.t_list_tr)
        else:
            fids = self.fid_list_te
            dict_ns = self.dict_fid_to_ns_te
            copy_t_list = copy.deepcopy(self.t_list_te)
        #
        
#         print(fids)
#         print(dict_ns)
#         print(copy_t_list)
            
            
        if tensor:
            t_list = [torch.tensor(ts).to(device) for ts in copy_t_list]
        else:
            t_list = copy_t_list
        #
        
        for fid in fids:
            ns = dict_ns[fid]
            Xs = dict_Xs[fid]
            ys = dict_ys[fid].reshape([ns, -1])
#             cprint('w', '--------------------')
#             cprint('w', fid)
#             cprint('w', ns)
#             cprint('w', Xs.shape)
#             cprint('w', ys.shape)
#             cprint('w', '--------------------')
            if scale:
                scaler_X = self.raw_dataset.dict_fid_to_scaler_X[fid]
                scaler_y = self.raw_dataset.dict_fid_to_scaler_y[fid]
                Xs = scaler_X.transform(Xs)
                ys = scaler_y.transform(ys)
            #
            
            if tensor:
                X_list[fid] = torch.tensor(Xs).to(device)
                y_list[fid] = torch.tensor(ys).to(device)
            else:
                X_list[fid] = Xs
                y_list[fid] = ys
                
#             cprint('r', Xs.mean(0))
#             cprint('r', ys.mean(0))
#             cprint('b', Xs.std(0))
#             cprint('b', ys.std(0))
#             cprint('g', Xs)
#             cprint('w', ys.shape)
        #
    
        return X_list, y_list, t_list
    
    
    def get_data_mfhogp(self, fold, train=True, scale=True):
        
        X_list, y_list, t_list = self.get_data(fold, train, scale, tensor=False)

        fids_list, fids_X, fids_y = [], [], []
        
        for fid_t in list(X_list.keys()):
            
            #print(fid_t)
            
            ns = y_list[fid_t].shape[0]
            ys = y_list[fid_t].reshape([ns, fid_t, fid_t])

            lf = fid_t
            hf = self.fid_max

            x_lf = np.linspace(0,1,lf)
            y_lf = np.linspace(0,1,lf)
            x_hf = np.linspace(0,1,hf)
            y_hf = np.linspace(0,1,hf)

            ys_interp = []

            for n in range(ns):
                u = ys[n]
                interp_fn = interpolate.interp2d(x_lf, y_lf, u, kind=self.interp)
                u_hf = interp_fn(x_hf, y_hf)
                ys_interp.append(np.expand_dims(u_hf, 0))
            # 

            ys_interp = np.concatenate(ys_interp)
            ys_interp_1d = ys_interp.reshape([ns, -1])
            #print(ys_interp_1d.shape)
            
            fids_list.append(fid_t)
            fids_X.append(X_list[fid_t])
            fids_y.append(ys_interp_1d)
        #
        
#         for fid, Xs, ys in zip(fids_list, fids_X, fids_y):
#             print(fid, Xs.shape, ys.shape)
#             cprint('r', Xs.mean(0))
#             cprint('r', ys.mean(0))
#             cprint('b', Xs.std(0))
#             cprint('b', ys.std(0))
#             cprint('r', Xs)
#             cprint('b', ys.shape)

        return fids_list, fids_X, fids_y

#     def get_data_sf(self, fold, train=True, scale=True):
#         if train:
#             fids_list, fids_X, fids_y = self.get_data_mfhogp(fold, train, scale)
#             X_lf = np.copy(fids_X[0])
#             y_lf = np.copy(fids_y[0])
#             for X, y in zip(fids_X, fids_y):
#                 ns = X.shape[0]
#                 X_lf[:ns, :] = X
#                 y_lf[:ns, :] = y
#             #
#             #err = np.sum(np.square(X_lf-fids_X[0]))
#             #print(err)
#             #err = np.sum(np.square(y_lf-fids_y[0]))
#             #print(err)
#             return X_lf, y_lf
#         else:
#             fids_list, fids_X, fids_y = self.get_data_mfhogp(fold, train, scale)
#             X_hf = np.copy(fids_X[-1])
#             y_hf = np.copy(fids_y[-1])
#             return X_hf, y_hf
#         #

    def get_data_sf(self, fold, train=True, scale=True):
        if train:
            fids_list, fids_X, fids_y = self.get_data_mfhogp(fold, train, scale)
            X_lf = np.vstack(fids_X)
            y_lf = np.vstack(fids_y)
            return X_lf, y_lf
        else:
            fids_list, fids_X, fids_y = self.get_data_mfhogp(fold, train, scale)
            X_hf = np.copy(fids_X[-1])
            y_hf = np.copy(fids_y[-1])
            return X_hf, y_hf
        #

#     def debug(self,):
#         X, y = self.get_data_sf(fold=0, train=False)
#         print(X.shape)
#         print(y.shape)
#         print(X)
        
#         fids_list, fids_X, fids_y = self.get_data_mfhogp(fold=0, train=False)
#         print(fids_X[0])
        
        
# dataset = MFData2D(
#      domain='Poisson',
#      fid_min=8,
#      fid_max=128,
#      t_min=0.0,
#      t_max=1.0,
#      fid_list_tr=[8, 48, 88, 128],
#      fid_list_te=[8, 48, 88, 128],
#      ns_list_tr=[50,20,20,5],
#      ns_list_te=[128,128,128,128],
# )

# dataset.debug()
        
  
#     def debug(self,):
#         self.get_data_mfhogp(fold=2)
#         self.get_data(fold=2)
        
        
# dataset = MFData2D(
#      domain='Poisson',
#      fid_min=8,
#      fid_max=128,
#      t_min=0.0,
#      t_max=1.0,
#      fid_list_tr=[8, 48, 88, 128],
#      fid_list_te=[8, 48, 88, 128],
#      ns_list_tr=[50, 20, 20, 5],
#      ns_list_te=[128,128,128,128],
# )

# dataset.debug()