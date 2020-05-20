import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import simulator as faas

from scipy import stats
from scipy.signal import find_peaks, peak_widths

##Simulator
faasSimulator = faas.faas_Model()

## repeating last fratio for short rows
def simulator(th, eps=[0]*94, seed=1):


    #eps.index = ['epsilon' + str(i) for i in np.arange(0,94)]
    #eps = [0] * 94

    if len(th) < 104:
        th = list(th) + []
        sims = faasSimulator.forward(np.concatenate([th, eps]), seed)
        return sims + np.random.normal(scale = 0.01, size=sims.shape)
        #mean of final ten entrie
    else:
        sims =  faasSimulator.forward(th, seed)
        return sims + np.random.normal(scale = 0.01, size=sims.shape)


def calc_summ(d):
    comp_d = []
    t = d['time']
    for i in d['data']:

        final = np.mean(i[-10:])
        #median of 3 highest points
        peak = np.max(i)
        #time to peak
        time_peak = t[np.argmax(i)]
        #sd = np.std(i[-10:])
        #time to final
        #time_final = np.argmax(np.logical_and((i[time_peak:] >= final - sd),(i[time_peak:] <= final + sd)) == True) + time_peak
        diff_peak_final = (final - peak)/peak * 100
        first = np.mean(i[:5])
        diff_peak_first = first - peak
        diff_first_final = first - final
        mid = i[len(i)//2]
        diff_first_mid = first - mid
        diff_final_mid = final - mid
        diff_peak_mid = peak - mid
        if time_peak > 259:
            time_peak = np.random.uniform(0, 259)
        if diff_peak_final > 25:
            diff_peak_final = np.random.uniform(0.1, 0.2)

        min_slope = np.diff(i).min()
        max_slope = np.diff(i[10:]).max()

        max_slope_index = t[np.argmax(np.diff(i[10:])) + 10]
        min_slope_index = t[np.argmin(np.diff(i[10:]))]

        #mean slope for first 10, 20, and then the remaining trace
        mean_10 = np.mean(np.diff(i)[:10])
        mean_30 = np.mean(np.diff(i)[10:30])
        mean_rest = np.mean(np.diff(i)[30:])

        #moments
        #mom_1 = stats.moment(i, 1, nan_policy = 'omit')
        mom_2 = stats.moment(i, 2, nan_policy = 'omit')
        mom_3 = stats.moment(i, 3, nan_policy = 'omit')
        mom_4 = stats.moment(i, 4, nan_policy = 'omit')
        mom_5 = stats.moment(i, 5, nan_policy = 'omit')

        #moments of differenced trace
        mom_diff_1 = stats.moment(np.diff(i), 1, nan_policy = 'omit')
        mom_diff_2 = stats.moment(np.diff(i), 2, nan_policy = 'omit')
        mom_diff_3 = stats.moment(np.diff(i), 3, nan_policy = 'omit')
        mom_diff_4 = stats.moment(np.diff(i), 4, nan_policy = 'omit')
        mom_diff_5 = stats.moment(np.diff(i), 5, nan_policy = 'omit')

        # peaks
        # peaks, _ = find_peaks(i)
        # peaks_widths = peak_widths(i, peaks)[0]

        comp_d.append([peak, final])# max_slope_index,
                      #min_slope_index, mean_10, mean_30, mean_rest, diff_first_mid, diff_final_mid,
                      #diff_peak_mid, mom_2, mom_3, mom_4, mom_5,
                      #mom_diff_1, mom_diff_2, mom_diff_3, mom_diff_4, mom_diff_5])
                      #peaks, peaks_widths])
        #comp_d.append([time_peak, min_slope, max_slope, diff_peak_final, diff_peak_first])# max_slope_index,
                      #min_slope_index, mean_10, mean_30, mean_rest, diff_first_mid, diff_final_mid,
                      #diff_peak_mid, mom_2, mom_3, mom_4, mom_5,
                      #mom_diff_1, mom_diff_2, mom_diff_3, mom_diff_4, mom_diff_5])
                      #peaks, peaks_widths])


    out = np.asarray(comp_d).flatten()
    #out = comp_d
    #out = d.flatten()
    if np.isnan(out).any() or np.isinf(out).any():
        print('Summary Failed!')
        return np.zeros(out.shape[0])
        #return d.flatten()

    #return out + np.random.rand(6*len(d))
    return out
