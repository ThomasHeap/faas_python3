import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from fratio import get_fratio_model
from multiprocessing import get_context, Pool
import torch

from scipy import stats
from scipy.signal import find_peaks, peak_widths




def sim(th, eps=[0]*94):

    theta = pd.Series(list(th))
    theta.index = ['logK_on_TN', 'logK_on_TC', 'logK_on_RN', 'logK_on_RC', 'logK_D_TN', 'logK_D_TC', 'logK_D_RN', 'logK_D_RC', 'm_alpha', 'alpha0'] + ['epsilon' + str(i) for i in np.arange(0,94)]
    x_model = get_fratio_model(theta)
    x = [list(x_model[i].iloc[:,1]) for i in range(len(x_model))]
    length = max(map(len, x))
    sims=np.array([xi+[xi[-1]]*(length-len(xi)) for xi in x])[:,1:]
    return sims


def simulator(th):
    th = th.numpy()
    if len(th.shape) < 2:
        th = th[np.newaxis,:]

    with get_context("forkserver").Pool() as pool:

        sims = pool.map(sim,th)
        pool.close()
        pool.join()

        return sims
