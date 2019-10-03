from fun_read_in import *

## get all experiment IDs
id_all = get_exp_id_ALL()

## read in all data
par_in = read_in('parameters', id_all)
par_in.columns = get_par_names()
timecourse_in = read_in('timecourse', id_all)
time_point_in = read_in('time_points', id_all).T
## merge all data into one dataframe
data_all = pd.concat([par_in, timecourse_in], axis=1)


def get_exp(cur_id):
    ## one specific experiment - determined in simulation script
    #cur_id = 1 # uncomment if experiment number not provided
    cur_id_name = id_all[cur_id]
    cur_par = par_in.iloc[[cur_id]]
    ## adujust measurement length
    cur_timecourse = timecourse_in.iloc[[cur_id]].T
    cur_time_point = time_point_in.iloc[0:len(cur_timecourse)]
    mask = np.array(pd.notna(cur_timecourse))
    return({'name':cur_id_name,'par':cur_par,'timecourse':cur_timecourse[mask],'time_point':cur_time_point[mask]})


def log10Kd_to_K(theta):
    # Ks = [K_on_TN =  pd.to_numeric(10^theta["logK_on_TN"]),
    #       K_on_TC =  pd.to_numeric(10^theta["logK_on_TC"]),
    #       K_on_RN =  pd.to_numeric(10^theta["logK_on_RN"]),
    #       K_on_RC =  pd.to_numeric(10^theta["logK_on_RC"]),
    #       K_off_TN = pd.to_numeric(10^(theta["logK_D_TN"] + theta["logK_on_TN"])),
    #       K_off_TC = pd.to_numeric(10^(theta["logK_D_TC"] + theta["logK_on_TC"])),
    #       K_off_RN = pd.to_numeric(10^(theta["logK_D_RN"] + theta["logK_on_RN"])),
    #       K_off_RC = pd.to_numeric(10^(theta["logK_D_RC"] + theta["logK_on_RC"]))]
    Ks = pd.Series([pd.to_numeric(10**theta["logK_on_TN"]),
          pd.to_numeric(10**theta["logK_on_TC"]),
          pd.to_numeric(10**theta["logK_on_RN"]),
          pd.to_numeric(10**theta["logK_on_RC"]),
          pd.to_numeric(10**(theta["logK_D_TN"] + theta["logK_on_TN"])),
          pd.to_numeric(10**(theta["logK_D_TC"] + theta["logK_on_TC"])),
          pd.to_numeric(10**(theta["logK_D_RN"] + theta["logK_on_RN"])),
          pd.to_numeric(10**(theta["logK_D_RC"] + theta["logK_on_RC"]))])

    Ks.index = ["K_on_TN", "K_on_TC", "K_on_RN", "K_on_RC", "K_off_TN", "K_off_TC", "K_off_RN",  "K_off_RC"]

    return(pd.concat([Ks,theta[np.setdiff1d(theta.index, ["logK_on_TN", "logK_on_TC", "logK_on_RN", "logK_on_RC", "logK_D_TN", "logK_D_TC", "logK_D_RN",  "logK_D_RC"])]]))

def K_to_log10Kd(theta):
    # logKs = [K_on_TN =  pd.to_numeric(np.log10(theta["logK_on_TN"])),
    #       K_on_TC =  pd.to_numeric(np.log10(theta["logK_on_TC"])),
    #       K_on_RN =  pd.to_numeric(np.log10(theta["logK_on_RN"])),
    #       K_on_RC =  pd.to_numeric(np.log10(theta["logK_on_RC"])),
    #       K_off_TN = pd.to_numeric(np.log10(theta["logK_D_TN"]/theta["logK_on_TN"])),
    #       K_off_TC = pd.to_numeric(np.log10(theta["logK_D_TC"]/theta["logK_on_TC"])),
    #       K_off_RN = pd.to_numeric(np.log10(theta["logK_D_RN"]/theta["logK_on_RN"])),
    #       K_off_RC = pd.to_numeric(np.log10(theta["logK_D_RC"]/theta["logK_on_RC"]))]
    logKs = pd.Series([pd.to_numeric(np.log10(theta["K_on_TN"])),
          pd.to_numeric(np.log10(theta["K_on_TC"])),
          pd.to_numeric(np.log10(theta["K_on_RN"])),
          pd.to_numeric(np.log10(theta["K_on_RC"])),
          pd.to_numeric(np.log10(theta["K_D_TN"]/theta["K_on_TN"])),
          pd.to_numeric(np.log10(theta["K_D_TC"]/theta["K_on_TC"])),
          pd.to_numeric(np.log10(theta["K_D_RN"]/theta["K_on_RN"])),
          pd.to_numeric(np.log10(theta["K_D_RC"]/theta["K_on_RC"]))])


    logKs.index = ["logK_on_TN", "logK_on_TC", "logK_on_RN", "logK_on_RC", "logK_off_TN", "logK_off_TC", "logK_off_RN",  "logK_off_RC"]
    return(pd.concat([logKs,theta[np.setdiff1d(theta.index, ["K_on_TN", "K_on_TC", "K_on_RN", "K_on_RC", "K_D_TN", "K_D_TC", "K_D_RN",  "K_D_RC"])]]))
