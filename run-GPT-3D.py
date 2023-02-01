import numpy as np
import random
import time
import fire

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from infras.randutils import *
from infras.misc import *
from infras.utils import *
from infras.configs import *

from core.inf_fid_gpt_3d import InfFidNet3D
from data.dataset3D import MFData3D
from data.domain_configs import EXP_DOMAIN_CONFIGS

from tqdm.auto import tqdm, trange

torch.set_default_tensor_type(torch.DoubleTensor)
        
def evaluate(**kwargs):

    exp_config = ExpConfigGPT()
    exp_config.parse(kwargs)
    
    device = torch.device(exp_config.device)
    domain = exp_config.domain

    dataset = MFData3DThreads(
         domain=domain,
         fid_min     = EXP_DOMAIN_CONFIGS[domain]['fid_min'],
         fid_max     = EXP_DOMAIN_CONFIGS[domain]['fid_max'],
         t_min       = EXP_DOMAIN_CONFIGS[domain]['t_min'],
         t_max       = EXP_DOMAIN_CONFIGS[domain]['t_max'],
         fid_list_tr = EXP_DOMAIN_CONFIGS[domain]['fid_list_tr'],
         fid_list_te = EXP_DOMAIN_CONFIGS[domain]['fid_list_te'],
         ns_list_tr  = EXP_DOMAIN_CONFIGS[domain]['ns_list_tr'],
         ns_list_te  = EXP_DOMAIN_CONFIGS[domain]['ns_list_te'],
    )
    
    exp_path = os.path.join(
        domain,
        'InfFidGPT',
        'base'+str(exp_config.h_dim),
        'fold'+str(exp_config.fold),
    )
    
    res_path = os.path.join('__res__', exp_path)
    log_path = os.path.join('__log__', exp_path)
    create_path(res_path)
    create_path(log_path)
    
    logger = get_logger(logpath=os.path.join(log_path, 'exp.log'), displaying=exp_config.verbose)
    logger.info(exp_config)
    
    perform_meters = PerformMeters(save_path=res_path, logger=logger)
    
    
    Xtr_list, ytr_list, t_list_tr = dataset.get_data(fold=exp_config.fold, train=True, device=device)

    Xte_list, yte_list, t_list_te = dataset.get_data(fold=exp_config.fold, train=False, device=device)
    
    inf_fid_model = InfFidNet3D(
        in_dim    = dataset.input_dim,
        h_dim     = exp_config.h_dim,
        s_dim     = dataset.fid_max,
        int_steps = exp_config.int_steps,
        solver    = exp_config.solver,
        dataset   = dataset,
        g_width   = exp_config.g_width,
        g_depth   = exp_config.g_depth,
        f_width   = exp_config.f_width,
        f_depth   = exp_config.f_depth,
        ode_t     = True,
        interp    = EXP_DOMAIN_CONFIGS[domain]['interp']
    ).to(device)
    
    
    max_epochs = exp_config.max_epochs

    optimizer = Adam(inf_fid_model.parameters(), lr=exp_config.max_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=exp_config.min_lr)
    
    
    for ie in trange(max_epochs+1):

        loss = inf_fid_model.eval_nelbo(Xtr_list, ytr_list, t_list_tr)

        if ie % exp_config.test_interval == 0:
            
            rmse_list_tr, adjust_rmse = inf_fid_model.eval_rmse(
                Xtr_list, ytr_list, t_list_tr, t_list_tr, return_adjust=True)
            
            rmse_list_te = inf_fid_model.eval_rmse(Xte_list, yte_list, t_list_te, t_list_tr)
            
            mae_list_tr = inf_fid_model.eval_mae(Xtr_list, ytr_list, t_list_tr, t_list_tr) 
            mae_list_te = inf_fid_model.eval_mae(Xte_list, yte_list, t_list_te, t_list_tr)
            
            pred_list = inf_fid_model.eval_pred(Xte_list, t_list_te, t_list_tr)
            
            perform_meters.update(
                ie, 
                loss.item(), 
                rmse_list_tr, 
                rmse_list_te, 
                mae_list_tr, 
                mae_list_te,
                pred_list
            )

            scheduler.step(adjust_rmse)

        #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        
if __name__=='__main__':
    
    fire.Fire(evaluate)



        