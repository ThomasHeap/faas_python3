from main import *
#from numba import njit, jit
from scipy.integrate import odeint
#import time

## one specific experiment - determine here for main script
## originally, one experiment for each batch was computed as
## the final pre-flash chemical concentrations were identical
## within one batch

## preflash ODEs
#@njit(parallel=True)
def f_preflash(y,t,params):
    CaDMn, DMn, Ca, OGB5N, CaOGB5N, NtNt, CtCt, CaNtNr, CaCtCr, CaNrCaNr, CaCrCaCr = y

    K_off_CaDMn = params[0]
    K_on_CaDMn = params[1]
    K_off_D = params[2]
    K_on_D = params[3]
    K_on_TN = params[4]
    K_on_TC = params[5]
    K_on_RN = params[6]
    K_on_RC = params[7]
    K_off_TN = params[8]
    K_off_TC = params[9]
    K_off_RN = params[10]
    K_off_RC = params[11]

    f = np.asarray([-K_off_CaDMn*CaDMn + K_on_CaDMn*DMn*Ca,       #CaDMn
          K_off_CaDMn*CaDMn - K_on_CaDMn*DMn*Ca,       #DMn
          K_off_CaDMn*CaDMn - K_on_CaDMn*DMn*Ca \
         -K_on_D*OGB5N*Ca + K_off_D*CaOGB5N \
       -2*K_on_TN*NtNt*Ca + K_off_TN*CaNtNr \
         -K_on_RN*CaNtNr*Ca + 2*K_off_RN*CaNrCaNr \
       -2*K_on_TC*CtCt*Ca + K_off_TC*CaCtCr \
         -K_on_RC*CaCtCr*Ca + 2*K_off_RC*CaCrCaCr,     #Ca
         -K_on_D*OGB5N*Ca + K_off_D*CaOGB5N,           #OGB5N
          K_on_D*OGB5N*Ca - K_off_D*CaOGB5N,           #CaOGB5N
       -2*K_on_TN*NtNt*Ca + K_off_TN*CaNtNr,           #NtNt
       -2*K_on_TC*CtCt*Ca + K_off_TC*CaCtCr,           #CtCt
        2*K_on_TN*NtNt*Ca - K_off_TN*CaNtNr \
         -K_on_RN*CaNtNr*Ca + 2*K_off_RN*CaNrCaNr,     #CaNtNr
        2*K_on_TC*CtCt*Ca - K_off_TC*CaCtCr \
         -K_on_RC*CaCtCr*Ca + 2*K_off_RC*CaCrCaCr,     #CaCtCr
          K_on_RN*CaNtNr*Ca - 2*K_off_RN*CaNrCaNr,     #CaNrCaNr
          K_on_RC*CaCtCr*Ca - 2*K_off_RC*CaCrCaCr      #CaCrCaCr
        ])
    return f

#@jit(parallel=True)
def f_preflash_jac(y,t,params):
    CaDMn, DMn, Ca, OGB5N, CaOGB5N, NtNt, CtCt, CaNtNr, CaCtCr, CaNrCaNr, CaCrCaCr = y

    K_off_CaDMn = params[0]
    K_on_CaDMn = params[1]
    K_off_D = params[2]
    K_on_D = params[3]
    K_on_TN = params[4]
    K_on_TC = params[5]
    K_on_RN = params[6]
    K_on_RC = params[7]
    K_off_TN = params[8]
    K_off_TC = params[9]
    K_off_RN = params[10]
    K_off_RC = params[11]

    f = np.array([[-K_off_CaDMn, Ca*K_on_CaDMn, DMn*K_on_CaDMn, 0, 0, 0, 0, 0, 0, 0, 0],
    [K_off_CaDMn, -Ca*K_on_CaDMn, -DMn*K_on_CaDMn, 0, 0, 0, 0, 0, 0, 0, 0],
    [K_off_CaDMn, -Ca*K_on_CaDMn, -CaCtCr*K_on_RC - CaNtNr*K_on_RN - 2*CtCt*K_on_TC - DMn*K_on_CaDMn - K_on_D*OGB5N - 2*K_on_TN*NtNt, -Ca*K_on_D, K_off_D, -2*Ca*K_on_TN, -2*Ca*K_on_TC, -Ca*K_on_RN + K_off_TN, -Ca*K_on_RC + K_off_TC, 2*K_off_RN, 2*K_off_RC],
    [0, 0, -K_on_D*OGB5N, -Ca*K_on_D, K_off_D, 0, 0, 0, 0, 0, 0],
    [0, 0, K_on_D*OGB5N, Ca*K_on_D, -K_off_D, 0, 0, 0, 0, 0, 0],
    [0, 0, -2*K_on_TN*NtNt, 0, 0, -2*Ca*K_on_TN, 0, K_off_TN, 0, 0, 0],
    [0, 0, -2*CtCt*K_on_TC, 0, 0, 0, -2*Ca*K_on_TC, 0, K_off_TC, 0, 0],
    [0, 0, -CaNtNr*K_on_RN + 2*K_on_TN*NtNt, 0, 0, 2*Ca*K_on_TN, 0, -Ca*K_on_RN - K_off_TN, 0, 2*K_off_RN, 0],
    [0, 0, -CaCtCr*K_on_RC + 2*CtCt*K_on_TC, 0, 0, 0, 2*Ca*K_on_TC, 0, -Ca*K_on_RC - K_off_TC, 0, 2*K_off_RC],
    [0, 0, CaNtNr*K_on_RN, 0, 0, 0, 0, Ca*K_on_RN, 0, -2*K_off_RN, 0],
    [0, 0, CaCtCr*K_on_RC, 0, 0, 0, 0, 0, Ca*K_on_RC, 0, -2*K_off_RC]])




    return f

## Compute sensitivity equations@njit
## Uncomment for hessian
## f_preflash_s <- sensitivitiesSymb(f_preflash) #FIND OUT HOW TO DO THIS IN PYTHON
## Generate ODE function
#func_preflash   <- funC(f_preflash, nGridpoints=0)
## func_preflash_s <- funC(c(f_preflash, f_preflash_s), nGridpoints=0)

def get_preflash_ss(theta, phi=get_exp(0)['par'], sensitivities=False):
    ## Paramters
    parms = pd.concat([phi["K_off_CaDMn"]*1000,
             phi["K_on_CaDMn"]*1000,
             phi["K_off_D"]*1000,
             phi["K_on_D"]*1000,
             theta])

    parms.index = ['K_off_CaDMn', 'K_on_CaDMn', 'K_off_D', 'K_on_D'] + list(theta.index)
    #print(parms)

    ## Calculation of initial Ca concentration
    DMn0 = float(phi["DM_tot"])
    Ca = float(phi["Ca_0"])
    OGB5N0 = float(phi["D_tot"])
    CaM0 = float(phi["B_tot"])
    K_D = float(parms["K_off_D"]/parms["K_on_D"])
    K_CaDMn = float(parms["K_off_CaDMn"]/parms["K_on_CaDMn"])
    K_TN = float(parms["K_off_TN"]/parms["K_on_TN"]/2)
    K_RN = float(2*parms["K_off_RN"]/parms["K_on_RN"])
    K_TC = float(parms["K_off_TC"]/parms["K_on_TC"]/2)
    K_RC = float(2*parms["K_off_RC"]/parms["K_on_RC"])

    Ca_initial = Ca + (Ca*DMn0)/(K_CaDMn + Ca) + (Ca*OGB5N0)/(K_D + Ca) + \
                 CaM0*(Ca*K_RN + 2*Ca**2)/(K_RN*K_TN + Ca*K_RN + Ca**2) + \
                 CaM0*(Ca*K_RC + 2*Ca**2)/(K_RC*K_TC + Ca*K_RC + Ca**2)

    #Ca_initial.columns = None

    ## Initial concentrations
    y = [0,            #CaDMn
         DMn0,         #DMn
         Ca_initial,   #Ca
         OGB5N0,       #OGB5N
         0,            #CaOGB5N
         CaM0,         #NtNt
         CaM0,         #CtCt
         0,            #CaNtNr
         0,            #CaCtCr
         0,            #CaNrCaNr
         0]            #CaCrCaCr

    #print(y)


    ## Run simulation for 10 seconds - equilibrium is certainly reached
    ## within this time
    times = np.arange(0,10,1)
    if sensitivities:
        out = odeint(f_preflash_s, y, times)
    else:
        #jac = lambda y, t: f_preflash_jac(y,t,parms.to_numpy())
        out = odeint(f_preflash, y, times, args = (parms.to_numpy(),), Dfun=f_preflash_jac)


    if ((np.isclose(out[-1,2], Ca)).all == True):
        print("Desired level of calcium", Ca, "not equal to actual level", out[-1,2])

    return(out[-1, :])
