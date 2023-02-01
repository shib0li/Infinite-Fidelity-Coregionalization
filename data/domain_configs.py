

INTERP_CONFIGS = {   
    'Poisson' : 'linear',
    'Heat' : 'linear',
    'Burgers' : 'cubic',
    'NavierStockURec' : 'cubic',
    'NavierStockPRec' : 'cubic',
    'TopOpt': 'cubic',
}

NS_STEPS = {   
    'NavierStockURec' : 2,
    'NavierStockVRec' : 2,
    'NavierStockPRec' : 2,
}

GEN_CONFIGS = {
    
    'Poisson' : {
        'fid_list_tr':[8, 16, 32, 64],
        'fid_list_te':[8, 16, 32, 64],
        'ns_list_tr':[128,128,128,128],
        'ns_list_te':[128,128,128,128],
        'n_threads':2,
    },
    
    'Heat' : {
        'fid_list_tr':[8, 16, 32, 64],
        'fid_list_te':[8, 16, 32, 64],
        'ns_list_tr':[128,128,128,128],
        'ns_list_te':[128,128,128,128],
        'n_threads':2,
    },
    
    'Burgers' : {
        'fid_list_tr':[8, 16, 32, 64],
        'fid_list_te':[8, 16, 32, 64],
        'ns_list_tr':[128,128,128,128],
        'ns_list_te':[128,128,128,128],
        'n_threads':2,
    },
    
    'TopOpt' : {
        'fid_list_tr':[50, 60, 70, 80, 90, 100, 110, 120],
        'fid_list_te':[50, 60, 70, 80, 90, 100, 110, 120],
        'ns_list_tr':[512]*8,
        'ns_list_te':[512]*8,
        'n_threads':4,
    },
    
    'NavierStockURec' : {
        'fid_list_tr':[16, 32, 48, 64, 80, 96],
        'fid_list_te':[16, 32, 48, 64, 80, 96],
        'ns_list_tr':[512]*6,
        'ns_list_te':[512]*6,
        'n_threads':4,
    },
    
    'NavierStockVRec' : {
        'fid_list_tr':[16, 32, 48, 64, 80, 96],
        'fid_list_te':[16, 32, 48, 64, 80, 96],
        'ns_list_tr':[512]*6,
        'ns_list_te':[512]*6,
        'n_threads':4,
    },
    
    'NavierStockPRec' : {
        'fid_list_tr':[16, 32, 48, 64, 80, 96],
        'fid_list_te':[16, 32, 48, 64, 80, 96],
        'ns_list_tr':[512]*6,
        'ns_list_te':[512]*6,
        'n_threads':4,
    },

}

EXP_DOMAIN_CONFIGS = {
   
    'Heat' : {
        'fid_min':8,
        'fid_max':64,
        't_min':0.0,
        't_max':1.0,
        'fid_list_tr':[8, 16, 32, 64],
        'ns_list_tr':[100, 50, 20, 5],
        'fid_list_te':[8, 16, 32, 64],
        'ns_list_te':[128,128,128,128],
        'interp':'bilinear',
    },

    'Poisson' : {
        'fid_min':8,
        'fid_max':64,
        't_min':0.0,
        't_max':1.0,
        'fid_list_tr':[8, 16, 32, 64],
        'ns_list_tr':[100, 50, 20, 5],
        'fid_list_te':[8, 16, 32, 64],
        'ns_list_te':[128,128,128,128],
        'interp':'bilinear',
    },
    
    'Burgers' : {
        'fid_min':16,
        'fid_max':64,
        't_min':0.0,
        't_max':1.0,
        'fid_list_tr':[16, 24, 32, 64],
        'ns_list_tr':[100, 50, 20, 5],
        'fid_list_te':[16, 24, 32, 64],
        'ns_list_te':[128,128,128,128],
        'interp':'bicubic',
    },
    
    'TopOpt' : {
        'fid_min':50,
        'fid_max':60,
        't_min':0.0,
        't_max':1.0,
        'fid_list_tr':[50, 60, 70, 80],
        'ns_list_tr':[256, 128, 64, 32],
        'fid_list_te':[50, 60, 70, 80],
        'ns_list_te':[256,256,256,256],
        'interp':'bicubic',
    },
    
    'NavierStockPRec' : {
        'fid_min':32,
        'fid_max':56,
        't_min':0.0,
        't_max':1.0,
        'fid_list_tr':[32, 48, 64, 80],
        'ns_list_tr':[256, 128, 64, 32],
        'fid_list_te':[32, 48, 64, 80],
        'ns_list_te':[128, 128, 128, 128],
        'interp':'bicubic',
    },

    'NavierStockURec' : {
        'fid_min':32,
        'fid_max':56,
        't_min':0.0,
        't_max':1.0,
        'fid_list_tr':[32, 48, 64, 80],
        'ns_list_tr':[256, 128, 64, 32],
        'fid_list_te':[32, 48, 64, 80],
        'ns_list_te':[128, 128, 128, 128],
        'interp':'bicubic',
    },

}