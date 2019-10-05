import rpy2
import rpy2.robjects as robjects
import numpy as np
import pandas

## import rpy2's package module
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('deSolve', 'cOde')

# R vector of strings
from rpy2.robjects import rinterface, r, IntVector, FloatVector, StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

robjects.r('''source("fratio.R")''')
robjects.r('''options(mc.cores=8)''')



class faas_Model():

    def __init__(self, faas_data_path = 'NULL'):


        # K_on_TN, K_on_TC, K_on_RN, K_on_RC, K_off_TN, K_off_TC, K_off_RN, K_off_RC, logK_on_TN, logK_on_TC, logK_on_RN, logK_on_RC, logK_D_TN, logK_D_TC, logK_D_RN, logK_D_RC, m_alpha, alpha0, epsilon0 - 93
        self.npar = 104
        #number of interesting params
        self.ipar = 10
        #number of nuisance params
        self.nuipar = 94

        self.theta_fiducial = np.concatenate((np.array([8.886491e+00,  7.924279e+00, 1.050515e+01,  7.397940e+00, -3.682371e+00, -4.509306e+00, -6.162727e+00, -6.585027e+00,  1.100000e-03, -3.900000e-01]), np.array([0]*94)))

        # Covariance matrix
        noise = 0.1
        self.C = np.diag([noise] * 94 * 3)
        self.Cinv = np.diag([1/noise] * 94 * 3)
        self.L = np.diag([1/(noise ** 0.5)] * 94 * 3)

        # Derivative of the covariance matrix
        self.n_sn = len(self.C)
        #self.dCdt = np.zeros((self.npar, self.n_sn, self.n_sn))

        # N data points
        self.ndata = 3*94


        self.time = np.genfromtxt('data/time_points.csv', delimiter=',')

        ## This gets a starting point for the parameters, excluding any noise term
        self.rget_theta0 = robjects.r['get_theta0']

        self.rget_fratio_exp = robjects.r['get_fratio_exp']
        self.rget_fratio_model = robjects.r['get_fratio_model']
        self.rget_exp = robjects.r['get_exp']
        self.theta0 = self.rget_theta0()

        # Compute the mean
        #self.mu = self.simulation(self.theta_fiducial, 0)

    # Generate realisation of \mu

    def forward(self, theta, seed):

        # Set the seed
        np.random.seed(seed)

        ## Gives a 94-element list of matrices with time and fratio columns -
        ## note that t=0 is present
        theta = FloatVector(theta)
        theta.names = self.theta0.names[8:]



        x_model = self.rget_fratio_model(theta)
        x = [list(x_model[i].rx(True,2)) for i in range(len(x_model))]
        length = max(map(len, x))
        fra=np.array([xi+[xi[-1]]*(length-len(xi)) for xi in x])[:,1:]

        # Noise
        #noise = np.random.normal(0, 0.01, fra.shape[1])
        #print(fra)
        return fra


    def compressor(self, d):
        comp_d = []

        for i in d:
            #mean of final ten entries
            final = np.mean(i[-10:])
            #median of 3 highest points
            peak = np.max(i)
            #time to peak
            time_peak = self.time[np.argmax(i)]
            #sd = np.std(i[-10:])
            #time to final
            #time_final = np.argmax(np.logical_and((i[time_peak:] >= final - sd),(i[time_peak:] <= final + sd)) == True) + time_peak
            #diff_peak_final = final - peak
            comp_d.append([final, peak, time_peak])# diff_peak_final])
        return np.asarray(comp_d).flatten()
    compressor_args=None

    def simulation(self, theta, seed):

        fra = self.compressor(self.forward(theta, seed))
        return fra

    def data(self):
        x_exp = self.rget_fratio_exp()

        x = [list(x_exp[i].rx(True,2)) for i in range(len(x_exp))]
        length = max(map(len, x))
        fra=np.array([xi+[xi[-1]]*(length-len(xi)) for xi in x])

        return fra

    def data_comp(self):

        fra = self.compressor(self.data())

        return fra

    def exp(self, id):
        return self.rget_exp(id)

    # Generate derivative of \mu w.r.t cosmological parameters
    def dmudt(self, theta_fiducial, h):

        # dmdt
        dmdt = np.zeros((self.npar, self.ndata))

        # Fiducial data
        d_fiducial = self.simulation(theta_fiducial, 0)

        # Loop over parameters
        for i in range(self.npar):
            # Step theta
            theta = np.copy(theta_fiducial)
            theta[i] += h[i]

            # Shifted data with same seed
            d_dash = self.simulation(theta, 0)

            # One step derivative
            #print((d_dash - d_fiducial)/h[i])
            dmdt[i,:] = (d_dash - d_fiducial)/h[i]

        #for i in range(self.nuipar):
        #    dmdt[i,:] =


        return dmdt
