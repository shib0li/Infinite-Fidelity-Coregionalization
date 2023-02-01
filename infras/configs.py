
class Config(object):
        
    def parse(self, kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        #
        
        print('=================================')
        print('*', self.config_name)
        print('---------------------------------')
        
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print('-', k, ':', getattr(self, k))
        print('=================================')
        
        
    def __str__(self,):
        
        buff = ""
        buff += '=================================\n'
        buff += ('*'+self.config_name+'\n')
        buff += '---------------------------------\n'
        
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                buff += ('-' + str(k) + ':' + str(getattr(self, k))+'\n')
            #
        #
        buff += '=================================\n'
        
        return buff    
    
    
class ExpConfigGPT(Config):

    domain = None
    fold = None
    
    h_dim=None 

    int_steps=2
    solver='dopri5'

    g_width=40
    g_depth=2
    
    f_width=40
    f_depth=2
    
    kernel='RBF'
    
    max_epochs=5000
    max_lr=1e-2
    min_lr=1e-3
    test_interval=100

    device = 'cpu'
    
    verbose=False
    
    def __init__(self,):
        super(ExpConfigGPT, self).__init__()
        self.config_name = 'Inf-Fid-GPT'
        
        
class ExpConfigODE(Config):

    domain = None
    fold = None
    
    h_dim=None 

    int_steps=2
    solver='dopri5'

    g_width=40
    g_depth=2
    
    f_width=40
    f_depth=2
    
    A_width=40
    A_depth=2
    
    max_epochs=5000
    max_lr=1e-2
    min_lr=1e-3
    test_interval=100

    device = 'cpu'
    
    verbose=False
    
    def __init__(self,):
        super(ExpConfigODE, self).__init__()
        self.config_name = 'Inf-Fid-ODE'
        
        
class ExpConfigSFid(Config):

    domain = None
    fold = None
    
    h_dim=None 

    g_width=40
    g_depth=2
    
    max_epochs=5000
    max_lr=1e-2
    min_lr=1e-3
    test_interval=100

    device = 'cpu'
    
    verbose=False
    
    def __init__(self,):
        super(ExpConfigSFid, self).__init__()
        self.config_name = 'Inf-Fid-ODE'
        
        
class ExpConfigDMFAL(Config):

    domain = None
    fold = None
    
    h_dim=None 
    
    hlayers_w = 40
    hlayers_d = 2
    
    
    max_epochs=5000
    lr=1e-3
    test_interval=100

    device = 'cpu'
    
    verbose=False
    
    def __init__(self,):
        super(ExpConfigDMFAL, self).__init__()
        self.config_name = 'DMFAL'
    

# class ConfigIFAL(Config):
    
#     device = 'cuda:0'
    
#     data_path = 'data/db'
    
#     #================================#
#     #        data set config
#     #================================#
#     domain = 'Poisson'
#     fid_min = 8
#     fid_max = 64
#     preload = 'data/db'
    
#     #================================#
#     #     inf fid surrogate config
#     #================================#
#     in_dim = 5
#     h_dim = 10  # dim of latent ode space
#     s_dim = 64  # dim of the pred soln space
    
#     int_steps=5
#     solver='rk4'
    
#     g_width=40
#     g_depth=1
#     f_width=40
#     f_depth=1
#     A_width=40
#     A_depth=1
    
#     #================================#
#     #     active learning config
#     #================================#
# #     burnin = 3
# #     nstotal = 5
# #     lr_init = 1e-3
# #     lr_stop = 2e-4
# #     L = 1
# #     nspred = 5
    
# #     heuristic = 'mi'
# #     acq_opt_steps = 2
# #     acq_opt_lr = 0.1
    
# #     max_active_steps = 10

#     burnin = 2000
#     nstotal = 50
#     lr_init = 1e-4
#     lr_stop = 2e-5
#     L = 10
#     nspred = 10

# #     burnin = 1
# #     nstotal = 5
# #     lr_init = 1e-4
# #     lr_stop = 2e-5
# #     L = 1
# #     nspred = 5
    
#     heuristic = 'mi'
#     acq_opt_steps = 20
#     acq_opt_lr = 0.1
    
#     max_active_steps = 500
    
    
#     #================================#
#     #           misc config
#     #================================#
#     verbose = False
    
#     def __init__(self,):
#         super(ConfigIFAL, self).__init__()
#         self.config_name = 'Infid AL Config'
        
        
        

        
        
        
        
     

        