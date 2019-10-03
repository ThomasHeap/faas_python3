import numpy as np
import pandas as pd

def read_in(myString, exp_id_all, dir='data'):
    data_in_misoriented = pd.read_csv(dir + '/' + str(myString) + '.csv', delimiter = ',', header=None)

    data_in = data_in_misoriented.T

    return data_in

def get_par_names():
    par_names = ['x','tau_f','tau_s','DM_tot','Kd_DM','K_on_CaDMn','K_off_CaDMn','Kd_PP','K_onPP','K_off_CaPP', 'D_tot','Kd_D','K_on_D','K_off_D','Ratio_D','Ca_0','B_tot','delay','alpha']

    return par_names

def get_exp_id_ALL():
    # experiment IDs
    exp_id_1021_WT = ['#1021_WT_360','#1021_WT_370','#1021_WT_380','#1021_WT_390','#1021_WT_410','#1021_WT_430','#1021_WT_440','#1021_WT_450','#1021_WT_460','#1021_WT_470','#1021_WT_480','#1021_WT_490','#1021_WT_500']

    exp_id_0611_Wta = ['#0611_WTa_380','#0611_WTa_400','#0611_WTa_420','#0611_WTa_425','#0611_WTa_435','#0611_WTa_440','#0611_WTa_445','#0611_WTa_455','#0611_WTa_460','#0611_WTa_480','#0611_WTa_500']

    exp_id_0611_Wtb = ['#0611_Wtb_380','#0611_Wtb_400','#0611_Wtb_410','#0611_Wtb_420','#0611_Wtb_425','#0611_Wtb_430','#0611_Wtb_440','#0611_Wtb_445','#0611_Wtb_460','#0611_Wtb_480','#0611_Wtb_500']

    exp_id_1107_Wta = ['#1107_Wta_360','#1107_Wta_370','#1107_Wta_380','#1107_Wta_390','#1107_Wta_400','#1107_Wta_410','#1107_Wta_420','#1107_Wta_430','#1107_Wta_440','#1107_Wta_450','#1107_Wta_460','#1107_Wta_470','#1107_Wta_480','#1107_Wta_490','#1107_Wta_500i','#1107_Wta_500ii']

    exp_id_1107_WTb = ['#1107_WTb_360','#1107_WTb_370','#1107_WTb_380','#1107_WTb_390','#1107_WTb_400','#1107_WTb_410','#1107_WTb_420','#1107_WTb_430','#1107_WTb_440','#1107_WTb_450','#1107_WTb_460','#1107_WTb_470','#1107_WTb_480','#1107_WTb_500']

    exp_id_1107_Wtc = ['#1107_WTc_360','#1107_WTc_370','#1107_WTc_380','#1107_WTc_390','#1107_WTc_400','#1107_WTc_410','#1107_WTc_420','#1107_WTc_430','#1107_WTc_440','#1107_WTc_450','#1107_WTc_460','#1107_WTc_470','#1107_WTc_480','#1107_WTc_490','#1107_WTc_500']

    exp_id_1107_Wtd = ['#1107_WTd_360','#1107_WTd_370','#1107_WTd_380','#1107_WTd_390','#1107_WTd_400','#1107_WTd_410','#1107_WTd_420','#1107_WTd_430','#1107_WTd_440','#1107_WTd_450','#1107_WTd_460','#1107_WTd_470','#1107_WTd_480','#1107_WTd_490']

    exp_id_all = exp_id_1021_WT+exp_id_0611_Wta+exp_id_0611_Wtb+exp_id_1107_Wta+exp_id_1107_WTb+exp_id_1107_Wtc+exp_id_1107_Wtd

    return exp_id_all
