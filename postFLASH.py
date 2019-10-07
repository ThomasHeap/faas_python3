from preFLASH import *
from numba import njit, jit
from scipy.integrate import odeint

## Equations
@njit
def f_postflash(y,t,params):
    CaDMn_s, CaDMn_f, DMn_s, DMn_f, CaPP, PP, Ca, OGB5N, CaOGB5N, NtNt, CtCt, CaNtNr, CaCtCr, CaNrCaNr, CaCrCaCr = y

    #K_off_CaDMn, K_on_CaDMn, K_off_D, K_on_D, K_off_CaPP, K_f, K_s, K_on_TN, K_on_TC, K_on_RN, K_on_RC, K_off_TN, K_off_TC, K_off_RN, K_off_RC = params
    K_off_CaDMn = params[0]
    K_on_CaDMn = params[1]
    K_off_D = params[2]
    K_on_D = params[3]
    K_off_CaPP = params[4]
    K_f = params[5]
    K_s = params[6]
    K_on_TN = params[7]
    K_on_TC = params[8]
    K_on_RN = params[9]
    K_on_RC = params[10]
    K_off_TN = params[11]
    K_off_TC = params[12]
    K_off_RN = params[13]
    K_off_RC = params[14]

    f = np.array([
      -K_off_CaDMn*CaDMn_s + K_on_CaDMn*DMn_s*Ca - CaDMn_s*K_s,                                     #CaDMn_s
      -K_off_CaDMn*CaDMn_f + K_on_CaDMn*DMn_f*Ca -  CaDMn_f*K_f,                                    #CaDMn_f
      K_off_CaDMn*CaDMn_s - K_on_CaDMn*DMn_s*Ca - DMn_s*K_s,                                        #DMn_s
      K_off_CaDMn*CaDMn_f - K_on_CaDMn*DMn_f*Ca - DMn_f*K_f,                                        #DMn_f
      -K_off_CaPP*CaPP + K_on_CaDMn*PP*Ca + CaDMn_s*K_s + CaDMn_f*K_f,                              #CaPP
      K_off_CaPP*CaPP - K_on_CaDMn*PP*Ca + 2*DMn_s*K_s + 2*DMn_f*K_f + CaDMn_s*K_s + CaDMn_f*K_f,   #PP
      K_off_CaDMn*CaDMn_s + K_off_CaDMn*CaDMn_f - K_on_CaDMn*DMn_s*Ca  - \
      K_on_CaDMn*DMn_f*Ca - K_on_D*OGB5N*Ca + K_off_D*CaOGB5N + \
      K_off_CaPP*CaPP - K_on_CaDMn*PP*Ca -2*K_on_TN*NtNt*Ca + \
      K_off_TN*CaNtNr -K_on_RN*CaNtNr*Ca + 2*K_off_RN*CaNrCaNr \
      -2*K_on_TC*CtCt*Ca + K_off_TC*CaCtCr -K_on_RC*CaCtCr*Ca + \
       2*K_off_RC*CaCrCaCr, # -K_on_CB*CB*Ca + K_off_CB*CaCB,                                        Ca
      - K_on_D*OGB5N*Ca + K_off_D*CaOGB5N,                                                          #OGB5N
      K_on_D*OGB5N*Ca - K_off_D*CaOGB5N,                                                            #CaOGB5N
      -2*K_on_TN*NtNt*Ca+K_off_TN*CaNtNr,                                                           #NtNt
      -2*K_on_TC*CtCt*Ca+K_off_TC*CaCtCr,                                                           #CtCt
      2*K_on_TN*NtNt*Ca-K_off_TN*CaNtNr-K_on_RN*CaNtNr*Ca+2*K_off_RN*CaNrCaNr,                      #CaNtNr
      2*K_on_TC*CtCt*Ca-K_off_TC*CaCtCr-K_on_RC*CaCtCr*Ca+2*K_off_RC*CaCrCaCr,                      #CaCtCr
      K_on_RN*CaNtNr*Ca-2*K_off_RN*CaNrCaNr,                                                        #CaNrCaNr
      K_on_RC*CaCtCr*Ca-2*K_off_RC*CaCrCaCr                                                         #CaCrCaCr
    ])

    return f


## Compute sensitivity equations
## Uncomment for hessian
#f_s <- sensitivitiesSymb(f)
## Generate ODE function
#func <- funC(f, nGridpoints=0)
#func_s <- funC(c(f, f_s), nGridpoints=0)

def postflash(theta=0, phi=get_exp(0)['par'], epsilon=0, time_points=0, hessian=False):
    ## Experimental - try to kill runaway computations after 1s - whether
    ## this works depends on if the cOde code checks for interrupts
    #setTimeLimit(cpu=5)

    ## all rate parameters
    parms = pd.concat([phi["K_off_CaDMn"]*1000,
                phi["K_on_CaDMn"]*1000,
                phi["K_off_D"]*1000,
                phi["K_on_D"]*1000,
                phi["K_off_CaPP"]*1000,
                1/(phi["tau_f"]/1000),      #Kf
                1/(phi["tau_s"]/1000),      #Ks
                theta])

    parms.index = ['K_off_CaDMn', 'K_on_CaDMn', 'K_off_D', 'K_on_D', 'K_off_CaPP', 'K_f', 'K_s'] + list(theta.index)
    ## uncaging and fast fraction

    alpha = max(float(1+epsilon) * float(phi['delay'] * theta['m_alpha'] + theta['alpha0']), 0)
    x = phi['x']

    ## same time points as experiment
    times = np.squeeze(np.asarray(time_points.T))/1000
    #times = np.arange(0,37, 0.01)



    ## read in pre-flash values and set concentrations
    pre_out = pd.Series(get_preflash_ss(theta, phi))

    pre_out.index = ['CaDMn', 'DMn', 'Ca', 'OGB5N', 'CaOGB5N', 'NtNt', 'CtCt', 'CaNtNr', 'CaCtCr', 'CaNrCaNr', 'CaCrCaCr']

    y = [
    float((1-x)*alpha*pre_out["CaDMn"]), #CaDMn_s
    float( x   *alpha*pre_out["CaDMn"]), #CaDMn_f
    float((1-x)*alpha*pre_out["DMn"]),   #DMn_s
    float( x   *alpha*pre_out["DMn"]),   #DMn_f
    0,                                             #CaPP
    0]                                             #PP
    y = y + list(pre_out[["Ca", "OGB5N", "CaOGB5N", "NtNt", "CtCt", "CaNtNr", "CaCtCr", "CaNrCaNr", "CaCrCaCr"]].to_list())
    ## solve the ODEs
    if (hessian):
        ## Not yet
        pass
    else:
        post_out = odeint(f_postflash, y, times, args = (parms.to_numpy(),), atol=1e-6, rtol=1e-6)

    ## Value of F_max/F_min
    F_ratio = phi['Ratio_D']
    ## time points to compare simulation and experiment at
    relevantID = np.arange(post_out.shape[0])
    #print(post_out[relevantID, 7])
    ## evaluate fluorescence time course
    ## create list of F-ratios
    F_ratio_course = (post_out[relevantID, 7] + float(F_ratio)*post_out[relevantID, 8])/(pre_out["OGB5N"] + float(F_ratio)*pre_out["CaOGB5N"])


    ## Missing some bits to do with the hessian here

    return(F_ratio_course)
