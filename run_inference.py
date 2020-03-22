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

            return {'data': states[:,::3], 'time': self.time}

    ## Summary Statistics

    #from delfi.summarystats.Score_Sum_Stats import Score_MLE_Projected
    #sim = Faas(simulator_args=simulator_args)
    #ndata = len(sim.gen_single(theta_f)['data'].flatten())
    #nuisance_indices = np.arange(10,104)
    #Score = Score_MLE_Projected(ndata = ndata, theta_fiducial=theta_f + [0.1]*94, nuisances=nuisance_indices, seed=0, n_summary=10)
    #Score.compute_mean_covariance(sim.gen_single, 50, simulator_args=simulator_args)
    #Score.compute_derivatives(sim.gen_single, 50, [0.01]*10 + [0.5]*94, simulator_args=simulator_args)
    #print('mu shape: ')
    #print(Score.mu.shape)
    #print('Cinv shape: ')
    #print(Score.Cinv.shape)
    #print('dmudt shape: ')
    #print(Score.dmudt.shape)
    #Score.compute_fisher()



    #np.save('mu.npy', Score.mu)
    #np.save('Cinv.npy', Score.Cinv)
    #np.save('F.npy', Score.F)
    #np.save('dmudt.npy', Score.dmudt)
    #np.save('dCdt.npy' , Score.dCdt)
    #print('Saved!')
    from delfi.summarystats.Score_Sum_Stats import Score_MLE_Projected

    mu = np.load('mu.npy')
    Cinv = np.load('Cinv.npy', allow_pickle=True)
    #Cinv = np.diag([10]*8178)
    F = np.load('F.npy', allow_pickle=True)
    dmudt = np.load('dmudt.npy', allow_pickle=True)
    dCdt = np.load('dCdt.npy', allow_pickle=True)

    sim = Faas(simulator_args=simulator_args)
    ndata = len(sim.gen_single(theta_f)['data'].flatten())
    nuisance_indices = np.arange(10,104)
    print(nuisance_indices)
    #print((theta_f + [0.1]*94)[np.delete(np.arange(len(theta_f + [0]*94)), nuisance_indices)])
    Score = Score_MLE_Projected(ndata = ndata, theta_fiducial=np.asarray(theta_f + [0.1]*94), nuisances=nuisance_indices, seed=0, mu=mu, Cinv=Cinv, dmudt=dmudt,F=F, n_summary = 10)
    ##Score.compute_mean_covariance(sim.gen_single, 10, simulator_args=simulator_args)
    ##Score.compute_derivatives(sim.gen_single, 10, [0.01]*10 + [0.5]*94, simulator_args=simulator_args)
    #print(Score.mu.shape)
    #print(Score.Cinv.shape)
    #print(Score.dmudt.shape)
    #print(Score.F.shape) 
    #Score.compute_fisher()

    ##Generator
    import delfi.generator as dg

    m = Faas(seed=0)
    s = Score
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
    n_rounds = 5

    # fitting setup
    minibatch = 100
    epochs = 100
    val_frac = 0.05

    # network setup
    n_hiddens = [50]*5

    # convenience
    prior_norm = True

    # MAF parameters
    density = 'maf'
    n_mades = 5       # number of MADES



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

