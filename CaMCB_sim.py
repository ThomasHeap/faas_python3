import fratio
import numpy as np
import pandas as pd

class CaMCB_Model():

    def __init__(self, faas_data_path = 'NULL'):


        self.theta_fiducial = np.array([8.886491e+00,  7.924279e+00, 1.050515e+01,  7.397940e+00, -3.682371e+00, -4.509306e+00, -6.162727e+00, -6.585027e+00,  1.100000e-03, -3.900000e-01])


        #self.rget_theta0 = robjects.r['get_theta0']
        #self.theta0 = theta0

        self.rget_camcb_model = fratio.get_camcb_model

    # Generate realisation of \mu

    def forward(self, th):

        # Set the seed

        ## Gives a 94-element list of matrices with time and fratio columns -
        ## note that t=0 is present
        theta = pd.Series(th)
        theta.index = ['logK_on_TN', 'logK_on_TC', 'logK_on_RN', 'logK_on_RC', 'logK_D_TN', 'logK_D_TC', 'logK_D_RN', 'logK_D_RC', 'm_alpha', 'alpha0']



        x_model = self.rget_camcb_model(theta=theta)
        #x = [list(x_model[i].rx(True,2)) for i in range(len(x_model))]
        #length = max(map(len, x))
        #fra=np.array([xi+[xi[-1]]*(length-len(xi)) for xi in x])[:,1:]

        # Noise
        #noise = np.random.normal(0, 0.01, fra.shape[1])
        #print(fra)
        return x_model
