import numpy as np
import os
import time

import fire
import json

from infras.configs import *
from data.rawdata import InfFidRawData

dict_domains = {
    
    'Heat' : {
        'fid_min':8,
        'fid_max':128,
        't_min':0.0,
        't_max':1.0,
        'fid_list_tr':[8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128],
        'ns_list_tr':[512]*16,
        'fid_list_te':[8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128],
        'ns_list_te':[512]*16,
    },
    
    'Poisson' : {
        'fid_min':8,
        'fid_max':128,
        't_min':0.0,
        't_max':1.0,
        'fid_list_tr':[8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128],
        'ns_list_tr':[512]*16,
        'fid_list_te':[8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128],
        'ns_list_te':[512]*16,
    },
    
    'Burgers' : {
        'fid_min':16,
        'fid_max':128,
        't_min':0.0,
        't_max':1.0,
        'fid_list_tr':[16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128],
        'ns_list_tr':[512]*15,
        'fid_list_te':[16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128],
        'ns_list_te':[512]*15,
    },
    
    'NavierStockURec' : {
        'fid_min':8,
        'fid_max':32,
        't_min':0.0,
        't_max':1.0,
        'fid_list_tr':[8, 16, 24, 32],
        'ns_list_tr':[64]*4,
        'fid_list_te':[8, 16, 24, 32],
        'ns_list_te':[64]*4,
    },

}

class GenDataConfig(Config):

    domain = None
    extend = True
    folder = '__raw__'
    
    def __init__(self,):
        super(GenDataConfig, self).__init__()
        self.config_name = 'Gen-DB-Config'

    
def generate(**kwargs):

    gen_config = GenDataConfig()
    gen_config.parse(kwargs)
    
    domain_config = dict_domains[gen_config.domain]
    
    dataset = InfFidRawData(
        domain=gen_config.domain, 
        fid_min=domain_config['fid_min'], 
        fid_max=domain_config['fid_max'], 
        fid_list_tr=domain_config['fid_list_tr'], 
        ns_list_tr=domain_config['ns_list_tr'], 
        fid_list_te=domain_config['fid_list_te'], 
        ns_list_te=domain_config['ns_list_te'], 
        preload=os.path.join('data', gen_config.folder),
    ) 
    
if __name__=='__main__':
    
    fire.Fire(generate)
    