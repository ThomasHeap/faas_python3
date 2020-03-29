import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import simulator as faas


if __name__ == "__main__":
    seed = 1

    ##Priors
    import delfi.distribution as dd

    upper = np.array([12,  10, 12,  10, -3, -4, -4,-4] + [0.1,0])
    lower = np.array([4,  2, 4, 2, -10, -10, -10, -12] + [-0.1,-0.5])

    prior_mean = np.array([0]*8 + [1.1e-03, -3.9e-01])
    prior_cov = np.diag(np.array([10]*8+[(0.0011*0.2)**2, (0.39*0.2)**2]))

    prior = dd.Uniform(lower= lower, upper=upper, seed=seed)

    epsilon_mean = np.array([0]*94)
    epsilon_cov = np.diag(np.array([0.5] * 94))
    epsilon_prior = dd.Gaussian(m=epsilon_mean, S=epsilon_cov, seed=seed)

    ##Simulator
    faasSimulator = faas.faas_Model()

    ## repeating last fratio for short rows
    def simulator(th, seed, simulator_args, batch):

        eps_prior = simulator_args[0]
        eps = eps_prior.gen()[0]
        #eps.index = ['epsilon' + str(i) for i in np.arange(0,94)]
        #eps = [0] * 94

        if len(th) < 104:
            return faasSimulator.forward(np.concatenate([th, eps]), seed)
        else:
            return faasSimulator.forward(th, seed)

    simulator_args = [epsilon_prior]
    theta0 = faasSimulator.theta0

    theta_f = [8.886491e+00,  7.924279e+00,
       1.050515e+01,  7.397940e+00, -3.682371e+00, -4.509306e+00, -6.162727e+00,
      -6.585027e+00,  1.100000e-03, -3.900000e-01]




    from delfi.simulator.BaseSimulator import BaseSimulator

    class Faas(BaseSimulator):
        def __init__(self, batch=0, simulator_args = simulator_args, seed=None):
            """Faas simulator

            Parameters
            ----------
            seed : int or None
                If set, randomness across runs is disabled
            """
            dim_param = 104

            super().__init__(dim_param=dim_param, seed=seed)
            self.batch = batch
            self.simulator_args = simulator_args
            self.seed = seed
            self.FaasSimulator = simulator
            self.time = np.genfromtxt('data/time_points.csv', delimiter=',')

        def gen_single(self, params):
            """Forward model for simulator for single parameter set

            Parametersnp.nanstd(stats, axis=0)
            ----------
            params : list or np.array, 1d of length dim_param
                Parameter vector

            Returns
            -------
            dict : dictionary with data
                The dictionary must contain a key data that contains the results of
                the forward run. Additional entries can be present.
            """
            params = np.asarray(params)

            assert params.ndim == 1, 'params.ndim must be 1'

            hh_seed = self.seed

            states = self.FaasSimulator(th = params, seed = seed, simulator_args = self.simulator_args, batch = self.batch)
            if np.isnan(states.flatten()).any() or np.isinf(states.flatten()).any():
                states = np.random.randn(94,259)
                    #return d.flatten()

            return {'data': states, 'time': self.time}

    ## Summary Statistics

class FaasStats(BaseSummaryStats):
    """Moment based SummaryStats class for the faas model

    Calculates summary statistics
    """
    def __init__(self, seed=None):
        """See SummaryStats.py for docstring"""
        super(FaasStats, self).__init__(seed=seed)
        self.time = np.genfromtxt('data/time_points.csv', delimiter=',')
    
    def compressor(self, d, t):
            comp_d = []
            
            out = d.flatten()
            if np.isnan(out).any() or np.isinf(out).any():
                return np.random.randn(out.shape[0])**2
                #return d.flatten()

            #return out + np.random.rand(6*len(d))
            return out
    
    def calc(self, repetition_list):
        """Calculate summary statistics

        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance

        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """
        stats = []
        for r in range(len(repetition_list)):
            x = repetition_list[r]

            N = x['data']
            t = self.time

            # concatenation of summary statistics
            sum_stats_vec = self.compressor(N, t)
            #sum_stats_vec = sum_stats_vec[0:self.n_summary]

            stats.append(sum_stats_vec)

        return np.asarray(stats)
    
    
    ##Generator
    import delfi.generator as dg
    
    #n_processes = 16
    #seeds_m = np.arange(1,n_processes+1,1)
    #m = []

    #for i in range(n_processes):
    #    m.append(Faas(seed=seeds_m[i]))
    m=Faas(seed=1)
    s = FaasStats(seed=1)
    g = dg.Default(model=m, prior=prior, summary=s)


    #Test with known parameters
    # known parameters and respective labels
    true_params = np.array([8.886491e+00,  7.924279e+00, 1.050515e+01,  7.397940e+00, 
                            -3.682371e+00, -4.509306e+00, -6.162727e+00, -6.585027e+00,  
                            1.100000e-03, -3.900000e-01])       
    labels_params = ['logKonTN', 'logKonTC', 'logKonRN', 'logKonRC', 'logKDTN', 'logKDTC', 'logKDRN', 'logKDRC', 'malpha', 'alpha0']


    # observed data: simulation given known parameters
    obs = m.gen_single(true_params)

    obs_stats = s.calc([obs])

    ##Inference


    seed_inf = 20

    pilot_samples = 20

    # training schedule
    n_train = 500
    n_rounds = 50

    # fitting setup
    minibatch = 100
    epochs = 100
    val_frac = 0.05

    # network setup
    n_hiddens = [100]*10

    # convenience
    prior_norm = True

    # MAF parameters
    density = 'maf'
    n_mades = 15       # number of MADES



    import delfi.inference as infer

    # inference object
    res = infer.SNPEC(g,
                    obs=obs_stats,
                    n_hiddens=n_hiddens,
                    seed=seed_inf,
                    pilot_samples=pilot_samples,
                    n_mades=n_mades,
                    prior_norm=prior_norm,
                    density=density, verbose=True)

    # train
    print('Training!')
    log, _, posterior = res.run(
                        n_train=n_train,
                        n_rounds=n_rounds,
                        minibatch=minibatch,
                        epochs=epochs,
                        silent_fail=False,
                        proposal='prior',
                        val_frac=val_frac,
                        verbose=True,)

    print('Sampling!')
    posterior_samples = posterior[0].gen(10000)
    np.save('posterior_samples.npy', posterior_samples)

