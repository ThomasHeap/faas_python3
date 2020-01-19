from main import *
#from numba import njit, jit
from scipy.integrate import odeint
#import time

## Equations
#@njit(parallel=True)
def f_postflash(y,t,params):
    Ca, NtNt, CtCt, CaNtNr, CaCtCr, CaNrCaNr, CaCrCaCr, CB, CaCB = y

    ##K_on_TN, K_on_TC, K_on_RN, K_on_RC, K_off_TN, K_off_TC, K_off_RN, K_off_RC = params
    K_on_TN = params[0]
    K_on_TC = params[1]
    K_on_RN = params[2]
    K_on_RC = params[3]
    K_off_TN = params[4]
    K_off_TC = params[5]
    K_off_RN = params[6]
    K_off_RC = params[7]
    K_on_CB = params[10]
    K_off_CB = params[11]

    f = np.asarray([
            -2*K_on_TN*NtNt*Ca   +   K_off_TN*CaNtNr
             -K_on_RN*CaNtNr*Ca + 2*K_off_RN*CaNrCaNr
           -2*K_on_TC*CtCt*Ca   +   K_off_TC*CaCtCr
             -K_on_RC*CaCtCr*Ca + 2*K_off_RC*CaCrCaCr
             -K_on_CB*CB*Ca     +   K_off_CB*CaCB,
             -2*K_on_TN*NtNt*Ca   +   K_off_TN*CaNtNr,
             -2*K_on_TC*CtCt*Ca   +   K_off_TC*CaCtCr,
             2*K_on_TN*NtNt*Ca   -   K_off_TN*CaNtNr
           -  K_on_RN*CaNtNr*Ca + 2*K_off_RN*CaNrCaNr,
           2*K_on_TC*CtCt*Ca   -   K_off_TC*CaCtCr
           -  K_on_RC*CaCtCr*Ca + 2*K_off_RC*CaCrCaCr,
           K_on_RN*CaNtNr*Ca - 2*K_off_RN*CaNrCaNr,
           K_on_RC*CaCtCr*Ca - 2*K_off_RC*CaCrCaCr,
           -K_on_CB*CB*Ca     +   K_off_CB*CaCB,
           K_on_CB*CB*Ca     -   K_off_CB*CaCB
    ])

    return f


def camcb(theta=0):
    ## Experimental - try to kill runaway computations after 1s - whether
    ## this works depends on if the cOde code checks for interrupts
    #setTimeLimit(cpu=5)

    ## all rate parameters
    parms = pd.concat([theta, pd.Series([7.5e7, 29.5])])

    parms.index = list(theta.index) + ['K_on_CB', 'K_off_CB']


    times = np.arange(0,0.01,0.0001)
    #times = np.arange(0,37, 0.01)

    y = [50*1E-6,
        100*1E-6,
        100*1E-6,
        0,
        0,
        0,
        0,
        120*1E-6,
        0]


    out = odeint(f_postflash, y, times, args = (parms.to_numpy(),))

    return(out)
