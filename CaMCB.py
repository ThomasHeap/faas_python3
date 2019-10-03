## Full CaM-CB binding system

from main import *
from scipy.integrate import odeint

## preflash ODEs
def f_camcb(y,t,params):
    Ca, NtNt, CtCt, CaNtNr, CaCtCr, CaNrCaNr, CaCrCaCr, CB, CaCB = y

    K_on_TN, K_on_TC, K_on_RN, K_on_RC, K_off_TN, K_off_TC, K_off_RN,  K_off_RC, K_on_CB, K_off_CB = params

    f = [Ca=    -2*K_on_TN*NtNt*Ca   +   K_off_TN*CaNtNr
                -K_on_RN*CaNtNr*Ca   + 2*K_off_RN*CaNrCaNr
                -2*K_on_TC*CtCt*Ca   +   K_off_TC*CaCtCr
                -K_on_RC*CaCtCr*Ca   + 2*K_off_RC*CaCrCaCr
                -K_on_CB*CB*Ca       +   K_off_CB*CaCB,
        NtNt=   -2*K_on_TN*NtNt*Ca   +   K_off_TN*CaNtNr,
        CtCt=   -2*K_on_TC*CtCt*Ca   +   K_off_TC*CaCtCr,
        CaNtNr=  2*K_on_TN*NtNt*Ca   -   K_off_TN*CaNtNr
                -K_on_RN*CaNtNr*Ca   + 2*K_off_RN*CaNrCaNr,
        CaCtCr=  2*K_on_TC*CtCt*Ca   -   K_off_TC*CaCtCr
                -  K_on_RC*CaCtCr*Ca + 2*K_off_RC*CaCrCaCr,
        CaNrCaNr=  K_on_RN*CaNtNr*Ca - 2*K_off_RN*CaNrCaNr,
        CaCrCaCr=  K_on_RC*CaCtCr*Ca - 2*K_off_RC*CaCrCaCr,
        CB=       -K_on_CB*CB*Ca     +   K_off_CB*CaCB,
        CaCB=      K_on_CB*CB*Ca     -   K_off_CB*CaCB
        ]

    return f

def get_camcb_model(theta=K_to_log10Kd([K_on_TN=7.7e+08,
                                                 K_on_TC=8.4e+7,
                                                 K_on_RN=3.2e+10,
                                                 K_on_RC=2.5e+7,
                                                 K_off_TN=1.6e+5,
                                                 K_off_TC=2.6e+3,
                                                 K_off_RN=2.2e+4,
                                                 K_off_RC=6.5]),
                    yini=[i*1e-6 for i in [Ca=50,
                           NtNt=100, CtCt=100, CaNtNr=0, CaCtCr=0, CaNrCaNr=0, CaCrCaCr=0,
                           =120, CaCB=0]])

    K_all = [log10Kd_to_K(theta),
             K_on_CB=7.5e+7,
             K_off_CB=29.5]

    times = np.arange(0,0.01,0.0001)
    out = odeint(f_camcb, y, times, args = (parms,))
    return out
