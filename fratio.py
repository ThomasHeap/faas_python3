from postFLASH import *
from multiprocessing import get_context, Pool
#set_start_method('forkserver')
from functools import partial
import numpy as np
import time


## Mapping of batc ids onto experiments
batch_tab = pd.concat([pd.DataFrame({'batch_id':1, 'expVec':np.arange(0,13), 'name':'1021_WT_360'}),
                       pd.DataFrame({'batch_id':2, 'expVec':np.arange(13,24), 'name':'0611_WTa_380'}),
                       pd.DataFrame({'batch_id':3, 'expVec':np.arange(24,35), 'name':'0611_WTb_380'}),
                       pd.DataFrame({'batch_id':4, 'expVec':np.arange(35,51), 'name':'1107_WTa_360'}),
                       pd.DataFrame({'batch_id':5, 'expVec':np.arange(51,65), 'name':'1107_WTb_360'}),
                       pd.DataFrame({'batch_id':6, 'expVec':np.arange(65,80), 'name':'1107_WTc_360'}),
                       pd.DataFrame({'batch_id':7, 'expVec':np.arange(80,94), 'name':'1107_WTd_360'}),
                       ])

## determine which batches shall be used
batch_ids = pd.unique(batch_tab['batch_id'])

def get_theta0(ids=batch_tab.loc[batch_tab.batch_id.isin(batch_ids)]['expVec']):
    th = pd.Series({'K_on_TN':7.7e+8, 'K_on_TC':8.4e+7, 'K_on_RN':3.2e+10, 'K_on_RC':2.5e+7, 'K_D_TN':1.6e+5, 'K_D_TC':2.6e+3, 'K_D_RN':2.2e+4, 'K_D_RC':6.5})
    th_logkd = K_to_log10Kd(th)
    theta = pd.Series({'logK_on_TN':th_logkd[0], 'logK_on_TC':th_logkd[1], 'logK_on_RN':th_logkd[2], 'logK_on_RC':th_logkd[3], 'logK_D_TN':th_logkd[4], 'logK_D_TC':th_logkd[5], 'logK_D_RN':th_logkd[6], 'logK_D_RC':th_logkd[7], 'm_alpha':0.0011, 'alpha0':-0.39})
    epsilons = pd.Series([0] * len(ids))
    epsilons.index = ['epsilon' + str(i) for i in ids]

    return (pd.concat([theta, epsilons]))

def get_fratio_exp(ids=batch_tab.loc[batch_tab.batch_id.isin(batch_ids)]['expVec']):
    ids = ids.reset_index(drop=True)
    func = lambda id : pd.concat([get_exp(id)['time_point'], get_exp(id)['timecourse']], ignore_index=True, axis=1)
    fratios = list(map(func, ids))

    for i in range(len(fratios)):
        fratios[i].columns = ['time', 'fratio']

    return fratios

def func(id,thetaf):
    cur = get_exp(id)
    F_ratio_course = postflash(thetaf, phi=cur['par'],
                               epsilon=thetaf[['epsilon' + str(id)]],
                               time_points = pd.concat([pd.Series([0]),cur['time_point']]))

    outp = pd.concat([pd.concat([pd.Series([0]),cur['time_point']], ignore_index=True), pd.Series(F_ratio_course)],axis=1)

    return(outp)

## iterate through all batch_ids
def get_fratio_model(theta, ids=batch_tab.loc[batch_tab.batch_id.isin(batch_ids)]['expVec']):
    with get_context("forkserver").Pool() as pool:
        ## determine the experiment IDs and the batch name
        ids = ids.reset_index(drop=True)
        #ids = [0]
        ## iterate through all experiments
        ## run script that computes post-flash simulation, the F-ratio and the sensitivity based Hessian
        theta = log10Kd_to_K(theta)
        functheta = partial(func, thetaf=theta)

        out = pool.map(functheta, ids)
        pool.close()
        pool.join()
        return out

#def plot_fratio(fratio, add=False):
## Not needed
if __name__ == '__main__':
    start = time.time()
    th = get_theta0()


    k = get_fratio_model(th)
    end = time.time()
    print(end - start)
    print(len(k))
