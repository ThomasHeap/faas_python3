from main import *
#from numba import jit
import re
from scipy.integrate import odeint

## one specific experiment - determine here for main script
## originally, one experiment for each batch was computed as
## the final pre-flash chemical concentrations were identical
## within one batch

## preflash ODEs
def f_preflash(y,t,params):
    CaDMn, DMn, Ca, OGB5N, CaOGB5N, NtNt, CtCt, CaNtNr, CaCtCr, CaNrCaNr, CaCrCaCr = y

    K_off_CaDMn = params['K_off_CaDMn']
    K_on_CaDMn = params['K_on_CaDMn']
    K_off_D = params['K_off_D']
    K_on_D = params['K_on_D']
    K_on_TN = params['K_on_TN']
    K_on_TC = params['K_on_TC']
    K_on_RN = params['K_on_RN']
    K_on_RC = params['K_on_RC']
    K_off_TN = params['K_off_TN']
    K_off_TC = params['K_off_TC']
    K_off_RN = params['K_off_RN']
    K_off_RC = params['K_off_RC']

    f = [-K_off_CaDMn*CaDMn + K_on_CaDMn*DMn*Ca,       #CaDMn
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
        ]
    return f

## Compute sensitivity equations
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
    DMn0 = phi["DM_tot"]
    Ca = phi["Ca_0"]
    OGB5N0 = phi["D_tot"]
    CaM0 = phi["B_tot"]
    K_D = parms["K_off_D"]/parms["K_on_D"]
    K_CaDMn =parms["K_off_CaDMn"]/parms["K_on_CaDMn"]
    K_TN = parms["K_off_TN"]/parms["K_on_TN"]/2
    K_RN = 2*parms["K_off_RN"]/parms["K_on_RN"]
    K_TC = parms["K_off_TC"]/parms["K_on_TC"]/2
    K_RC = 2*parms["K_off_RC"]/parms["K_on_RC"]

    Ca_initial = Ca + (Ca*DMn0)/(K_CaDMn + Ca) + (Ca*OGB5N0)/(K_D + Ca) + \
                 CaM0*(Ca*K_RN + 2*Ca**2)/(K_RN*K_TN + Ca*K_RN + Ca**2) + \
                 CaM0*(Ca*K_RC + 2*Ca**2)/(K_RC*K_TC + Ca*K_RC + Ca**2)

    Ca_initial.columns = None

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


    ## Run simulation for 10 seconds - equilibrium is certainly reached
    ## within this time
    times = np.arange(0,10,1)
    if sensitivities:
        out = odeint(f_preflash_s, y, times)
    else:
        out = odeint(f_preflash, y, times, args = (parms,))

    if ((np.isclose(out[-1,2], Ca)).all == True):
        print("Desired level of calcium", Ca, "not equal to actual level", out[-1,2])

    return(out[-1, :])
