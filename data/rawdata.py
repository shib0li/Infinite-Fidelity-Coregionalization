import numpy as np
import copy
import random

import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from data.pde_solvers import *
from infras.randutils import *
from infras.misc import *

import pickle
from tqdm.auto import tqdm, trange
from sklearn.model_selection import KFold

class InfFidRawData:
    def __init__(self,
                 domain,
                 fid_min,
                 fid_max,
                 t_min=0.0,
                 t_max=1.0,
                 fid_list_tr=None,
                 fid_list_te=None,
                 ns_list_tr=None,
                 ns_list_te=None,
                 preload=None,
            ):
        
        if domain == 'Poisson':
            self.solver = Poisson()
        elif domain == 'Heat':
            self.solver = Heat()
        elif domain == 'Burgers':
            self.solver = Burgers()
        elif domain == 'TopOpt':
            self.solver = TopOpt()
        elif domain == 'NavierStock':
            self.solver = NavierStock()
        elif domain == 'NavierStockPRec':
            self.solver = NavierStockPRec()
        elif domain == 'NavierStockURec':
            self.solver = NavierStockURec()
        elif domain == 'NavierStockVRec':
            self.solver = NavierStockVRec()
        else:
            raise Exception('No PDE solvers found...')
        #
        
        assert self.solver.lb.size == self.solver.ub.size
        self.input_dim = self.solver.lb.size
        
        if preload is None:
            raise Exception('Does not specify a path to load or save the data')
        
        data_path = os.path.join(preload, domain+'_'+str(fid_min)+'_'+str(fid_max))
        
        if os.path.isdir(data_path):
            cprint('g', 'Existing raw data FOUND.')
            self._load_preload(data_path)
        else:
            cprint('r', 'WARNING: no existing raw data...')
            cprint('w', '  preparing new data will take extreme long time')
            
            create_path(data_path)

            self._init_mappings(t_min, t_max, fid_min, fid_max)

            self.fid_list_tr = copy.deepcopy(fid_list_tr)
            self.fid_list_te = copy.deepcopy(fid_list_te)

            self.fid_list = list(set(self.fid_list_tr + self.fid_list_te))
            self.fid_list.sort()

            self.ns_list_tr = copy.deepcopy(ns_list_tr)
            self.ns_list_te = copy.deepcopy(ns_list_te)

            self.t_list_tr = [self.func_fid_to_t(fid) for fid in self.fid_list_tr]
            self.t_list_te = [self.func_fid_to_t(fid) for fid in self.fid_list_te]

            #print(self.t_list_tr)
            #print(self.t_list_te)

            assert len(self.fid_list_tr) == len(self.ns_list_tr)
            assert len(self.fid_list_te) == len(self.ns_list_te)

            #self.nfids = int(fid_max-fid_min+1)
            #self.nfids_tr = len(self.fid_list_tr)
            #self.nfids_te = len(self.fid_list_te)

            self.dict_fid_to_ns_tr = {}
            self.dict_fid_to_ns_te = {}
            for fid, ns in zip(self.fid_list_tr, self.ns_list_tr):
                self.dict_fid_to_ns_tr[fid] = ns
            for fid, ns in zip(self.fid_list_te, self.ns_list_te):
                self.dict_fid_to_ns_te[fid] = ns

            self.dict_fid_to_Xtr, self.dict_fid_to_ytr = \
                self._get_fids_solns(
                    self.fid_list_tr, self.ns_list_tr, init_method='lhs'
                )


            self.dict_fid_to_Xte, self.dict_fid_to_yte = \
                self._get_fids_solns(
                    self.fid_list_te, self.ns_list_te, init_method='uniform'
                )

            self.dict_fid_to_scaler_X, self.dict_fid_to_scaler_y = self._eval_fids_info()

            self._save_preload(data_path)

            
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
            #cprint('r', '{:3f}-{}-{}'.format(t, fid, self.fid_list[idx]))
            err_t = np.abs(t-t_rev)
            #cprint('b', '{:.5f}-{:.5f}-{:.5f}'.format(t, t_rev, err_t))
            if fid != fid_list_all[idx]:
                raise Exception('Check the mappings of fids')
            #
            if err_t >= (t_max-t_min)/(fid_max-fid_min):
                raise Exception('Check the mappings of t')
            #
        #
        
        
    def _get_fids_solns(self, fid_list, ns_list, init_method, seed=0):
        
        dict_Xs = {}
        dict_ys = {}
        
        nfids = len(fid_list)
        
        for ifid in trange(nfids, desc='Gen Solns'):
            fid = fid_list[ifid]
            ns = ns_list[ifid]
            Xs = generate_with_bounds(
                N=ns, 
                lb=self.solver.lb, 
                ub=self.solver.ub, 
                method=init_method, 
                #seed=seed+ifid
                seed=seed
            )
            ys = self.solver.solve(Xs, fid)
            dict_Xs[fid] = Xs
            dict_ys[fid] = ys
        #
        
        return dict_Xs, dict_ys
    
    
    def _eval_fids_info(self,):

        dict_fid_to_scaler_X = {}
        dict_fid_to_scaler_y = {}
        
        for fid in self.fid_list:
            if fid in self.fid_list_tr and fid in self.fid_list_te:
                #cprint('r', fid)
                Xtr = self.dict_fid_to_Xtr[fid]
                ytr = self.dict_fid_to_ytr[fid]
                Xte = self.dict_fid_to_Xte[fid]
                yte = self.dict_fid_to_yte[fid]
                
                Xs = np.vstack([Xtr, Xte])
                ys = np.vstack([ytr, yte])
                
            elif fid in self.fid_list_tr and fid not in self.fid_list_te: 
                #cprint('b', fid)
                Xs = self.dict_fid_to_Xtr[fid]
                ys = self.dict_fid_to_ytr[fid]
                
            elif fid not in self.fid_list_tr and fid in self.fid_list_te:
                #cprint('g', fid)
                Xs = self.dict_fid_to_Xte[fid]
                ys = self.dict_fid_to_yte[fid]
            else:
                raise Exception('Invalid fidelity')
                
            N = ys.shape[0]
            re_ys = ys.reshape([N, -1])
                
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            scaler_X.fit(Xs)
            scaler_y.fit(re_ys)
            
            dict_fid_to_scaler_X[fid] = scaler_X
            dict_fid_to_scaler_y[fid] = scaler_y

        #
        
        return dict_fid_to_scaler_X, dict_fid_to_scaler_y
    
    
    def _save_preload(self, save_path):

        cat = {}
        cat['fid_min'] = self.fid_min
        cat['fid_max'] = self.fid_max
        cat['t_min'] = self.t_min
        cat['t_max'] = self.t_max

        cat['train'] = {}
        cat['test'] = {}

        cat['train']['t_list'] = self.t_list_tr
        cat['train']['fid_list'] = self.fid_list_tr
        cat['train']['ns_list'] = self.ns_list_tr

        cat['test']['t_list'] = self.t_list_te
        cat['test']['fid_list'] = self.fid_list_te
        cat['test']['ns_list'] = self.ns_list_te
        
        with open(os.path.join(save_path, 'cat.pkl'), 'wb') as handle:
            pickle.dump(cat, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        for fid in self.fid_list_tr:
            #cprint('r', fid)
            mat_path = os.path.join(save_path, 'train', 'fidelity_'+str(fid))
            create_path(mat_path)
            np.save(os.path.join(mat_path, 'Xs'), self.dict_fid_to_Xtr[fid])
            np.save(os.path.join(mat_path, 'ys'), self.dict_fid_to_ytr[fid])
            
        for fid in self.fid_list_te:
            #cprint('b', fid)
            mat_path = os.path.join(save_path, 'test', 'fidelity_'+str(fid))
            create_path(mat_path)
            np.save(os.path.join(mat_path, 'Xs'), self.dict_fid_to_Xte[fid])
            np.save(os.path.join(mat_path, 'ys'), self.dict_fid_to_yte[fid])
            
        for fid in self.fid_list:
            scaler_path = os.path.join(save_path, 'scalers', 'fidelity_'+str(fid))
            create_path(scaler_path)
            with open(os.path.join(scaler_path,'scaler_Xs.pkl'),'wb') as handle:      
                pickle.dump(
                    self.dict_fid_to_scaler_X[fid], 
                    handle, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )
            with open(os.path.join(scaler_path,'scaler_ys.pkl'),'wb') as handle:      
                pickle.dump(
                    self.dict_fid_to_scaler_y[fid], 
                    handle, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )
    
    def _load_preload(self, save_path):
        with open(os.path.join(save_path, 'cat.pkl'), 'rb') as handle:
            cat = pickle.load(handle)
        #
        #print(cat.keys())
        #print(cat['dict_fid_to_cost'])

        #self.fid_min = cat['fid_min']
        #self.fid_max = cat['fid_max']
        #self.t_min = cat['t_min']
        #self.t_max = cat['t_max']

        self._init_mappings(
            t_min=cat['t_min'], 
            t_max=cat['t_max'], 
            fid_min=cat['fid_min'], 
            fid_max=cat['fid_max']
        )
        
        self.fid_list_tr = cat['train']['fid_list']
        self.fid_list_te = cat['test']['fid_list']
        
        self.fid_list = list(set(self.fid_list_tr + self.fid_list_te))
        self.fid_list.sort()
        
        self.ns_list_tr = cat['train']['ns_list']
        self.ns_list_te = cat['test']['ns_list']
        
        self.t_list_tr = cat['train']['t_list']
        self.t_list_te = cat['test']['t_list']
    
        assert len(self.fid_list_tr) == len(self.ns_list_tr)
        assert len(self.fid_list_te) == len(self.ns_list_te)
        
        #self.nfids = int(self.fid_max-self.fid_min+1)
        #self.nfids_tr = len(self.fid_list_tr)
        #self.nfids_te = len(self.fid_list_te)
        
        self.dict_fid_to_ns_tr = {}
        self.dict_fid_to_ns_te = {}
        for fid, ns in zip(self.fid_list_tr, self.ns_list_tr):
            self.dict_fid_to_ns_tr[fid] = ns
        for fid, ns in zip(self.fid_list_te, self.ns_list_te):
            self.dict_fid_to_ns_te[fid] = ns
            
        #cprint('g', self.dict_fid_to_ns_tr)
        #cprint('g', self.dict_fid_to_ns_te)
        
        self.dict_fid_to_Xtr = {} 
        self.dict_fid_to_ytr = {}
        
        for fid in self.fid_list_tr:
            mat_path = os.path.join(save_path, 'train', 'fidelity_'+str(fid))
            Xs = np.load(os.path.join(mat_path, 'Xs.npy'))
            ys = np.load(os.path.join(mat_path, 'ys.npy'))
            #cprint('r', '{}-{}'.format(Xs.shape, ys.shape))
            self.dict_fid_to_Xtr[fid] = Xs
            self.dict_fid_to_ytr[fid] = ys
        #
        
        self.dict_fid_to_Xte = {} 
        self.dict_fid_to_yte = {}
        
        for fid in self.fid_list_te:
            mat_path = os.path.join(save_path, 'test', 'fidelity_'+str(fid))
            #print(mat_path)
            Xs = np.load(os.path.join(mat_path, 'Xs.npy'))
            ys = np.load(os.path.join(mat_path, 'ys.npy'))
            #cprint('b', '{}-{}'.format(Xs.shape, ys.shape))
            self.dict_fid_to_Xte[fid] = Xs
            self.dict_fid_to_yte[fid] = ys
        #   
        
        self.dict_fid_to_scaler_X = {}
        self.dict_fid_to_scaler_y = {}
        
        for fid in self.fid_list:
            scaler_path = os.path.join(save_path, 'scalers', 'fidelity_'+str(fid))
            with open(os.path.join(scaler_path,'scaler_Xs.pkl'),'rb') as handle:      
                scaler_Xs = pickle.load(handle)
            with open(os.path.join(scaler_path,'scaler_ys.pkl'),'rb') as handle:      
                scaler_ys = pickle.load(handle)
            #
            self.dict_fid_to_scaler_X[fid] = scaler_Xs
            self.dict_fid_to_scaler_y[fid] = scaler_ys
        #
        
        