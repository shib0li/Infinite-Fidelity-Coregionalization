
import os, sys
import pickle
import logging
import numpy as np


def get_logger(logpath, displaying=True, saving=True, debug=False, append=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        if append:
            info_file_handler = logging.FileHandler(logpath, mode="a")
        else:
            info_file_handler = logging.FileHandler(logpath, mode="w+")
        #
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

class PerformMeters(object):
    
    def __init__(self, save_path, logger=None):
    
        self.save_path = save_path
        self.logger = logger
        
        self.hist_rmse_list_tr = {}
        self.hist_rmse_list_te = {}
        
        self.hist_mae_list_tr = {}
        self.hist_mae_list_te = {}
        
        self.epochs_cnt = 0
        self.best_rmse_hf = np.inf
        self.best_pred_list = None
        self.best_epoch = 0
        
    def _dump_meter(self,):
        
        res = {}
        
        res['hist_rmse_list_tr'] = self.hist_rmse_list_tr
        res['hist_rmse_list_te'] = self.hist_rmse_list_te
        
        res['hist_mae_list_tr'] = self.hist_mae_list_tr
        res['hist_mae_list_te'] = self.hist_mae_list_te
        
        error_fname = 'error_epoch'+str(self.epochs_cnt)+'.pickle'

        with open(os.path.join(self.save_path, error_fname), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        
        ### dump best_pred
        best_pred = {}
        best_pred['rmse'] = self.best_rmse_hf
        best_pred['epoch'] = self.best_epoch
        best_pred['pred_list'] = self.best_pred_list
        
        with open(os.path.join(self.save_path, 'best_pred_list.pickle'), 'wb') as handle:
            pickle.dump(best_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        
    def update(self, 
               epoch, 
               loss,
               rmse_list_tr,
               rmse_list_te,
               mae_list_tr,
               mae_list_te,
               pred_list_te,
              ):
        
        self.hist_rmse_list_tr[epoch] = rmse_list_tr
        self.hist_rmse_list_te[epoch] = rmse_list_te
        
        self.hist_mae_list_tr[epoch] = mae_list_tr
        self.hist_mae_list_te[epoch] = mae_list_te
        
        fids_list_tr = list(rmse_list_tr.keys())
        fids_list_te = list(rmse_list_tr.keys())
        
        rmse_hf = rmse_list_te[fids_list_te[-1]]
        if rmse_hf < self.best_rmse_hf:
            self.best_rmse_hf = rmse_hf
            self.best_pred_list = pred_list_te
            self.best_epoch = epoch
        #
        
        
        if self.logger is not None:    
            self.logger.info('=========================================')
            self.logger.info('             Epochs {} '.format(epoch))
            self.logger.info('=========================================') 
            
            self.logger.info('\n### Loss={:.5f} ###'.format(loss))
            
            self.logger.info('\n### Eval train examples ###')
            for fid in fids_list_tr:
                self.logger.info('    - fid={}, rmse={:.5f}, mae={:.5f}'.format(
                    fid, rmse_list_tr[fid], mae_list_tr[fid]))
            #
            
            self.logger.info('\n### Eval test examples ###')
            for fid in fids_list_te:
                self.logger.info('    - fid={}, rmse={:.5f}, mae={:.5f}'.format(
                    fid, rmse_list_te[fid], mae_list_te[fid]))
            #
        #
        
        self._dump_meter()
        
        self.epochs_cnt += 1
   
        #
    
    
class PerformMetersSF(object):
    
    def __init__(self, save_path, logger=None):
    
        self.save_path = save_path
        self.logger = logger
        
        self.hist_rmse_tr = {}
        self.hist_rmse_te = {}
        
        self.hist_mae_tr = {}
        self.hist_mae_te = {}
        
        self.epochs_cnt = 0
        self.best_rmse = np.inf
        self.best_pred = None
        self.best_epoch = 0
        
    def _dump_meter(self,):
        
        res = {}
        
        res['hist_rmse_tr'] = self.hist_rmse_tr
        res['hist_rmse_te'] = self.hist_rmse_te
        
        res['hist_mae_tr'] = self.hist_mae_tr
        res['hist_mae_te'] = self.hist_mae_te
        
        error_fname = 'error_epoch'+str(self.epochs_cnt)+'.pickle'

        with open(os.path.join(self.save_path, error_fname), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        
        ### dump best_pred
        best_pred = {}
        best_pred['rmse'] = self.best_rmse
        best_pred['epoch'] = self.best_epoch
        best_pred['pred'] = self.best_pred
        
        with open(os.path.join(self.save_path, 'best_pred.pickle'), 'wb') as handle:
            pickle.dump(best_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        
    def update(self, 
               epoch, 
               loss,
               rmse_tr,
               rmse_te,
               mae_tr,
               mae_te,
               pred_te,
              ):
        
        self.hist_rmse_tr[epoch] = rmse_tr
        self.hist_rmse_te[epoch] = rmse_te
        
        self.hist_mae_tr[epoch] = mae_tr
        self.hist_mae_te[epoch] = mae_te
        
#         fids_list_tr = list(rmse_list_tr.keys())
#         fids_list_te = list(rmse_list_tr.keys())
        
        if rmse_te < self.best_rmse:
            self.best_rmse = rmse_te
            self.best_pred = pred_te
            self.best_epoch = epoch
        #
        
        
        if self.logger is not None:    
            self.logger.info('=========================================')
            self.logger.info('             Epochs {} '.format(epoch))
            self.logger.info('=========================================') 
            
            self.logger.info('\n### Loss={:.5f} ###'.format(loss))
            
            self.logger.info('\n### Eval train examples ###')
            self.logger.info('rmse={:.5f}, mae={:.5f}'.format(rmse_tr, mae_tr))
            
            self.logger.info('\n### Eval test examples ###')
            self.logger.info('rmse={:.5f}, mae={:.5f}'.format(rmse_te, mae_te))
        #
        
        self._dump_meter()
        
        self.epochs_cnt += 1
   
        #   
        
