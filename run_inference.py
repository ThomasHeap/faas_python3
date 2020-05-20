# MATH
import numpy as np
import torch

# VISUALIZATION
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
import sbi
import sbi.utils as utils
from sbi.inference.snpe.snpe_c import SnpeC

from faas_helper_fns import simulator, calc_summ

def run_faas_model(params):

    params = np.asarray(params)

    states = simulator(th=params, seed=1)
    t = np.genfromtxt('data/time_points.csv', delimiter=',')

    return {'data': states,
            'time': t}

def simulation_wrapper(params):
    """
    Takes in conductance values and then first runs the Hodgkin Huxley model and then returns the summary statistics as torch.Tensor
    """
    obs = run_faas_model(params)
    summstats = torch.as_tensor(calc_summ(d=obs))
    return summstats

if __name__ == "__main__":
    prior_max = np.array([12,  10, 12,  10, -3, -4, -4,-4] + [0.05,-0.5])
    prior_min = np.array([4,  2, 4, 2, -10, -10, -10, -12] + [-0.05,-0.5])
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

        # true parameters and respective labels
    true_params = np.array([8.886491e+00,  7.924279e+00,
       1.050515e+01,  7.397940e+00, -3.682371e+00, -4.509306e+00, -6.162727e+00,
<<<<<<< HEAD
      -6.585027e+00,  1.100000e-03, -3.900000e-01])
    labels_params = ['KonTN', 'KonTC', 'KonRN', 'KonRC', 'KDTN', 'KDTC', 'KDRN', 'KDRC', 'malpha', 'alpha0']
=======
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
    #dCdt = np.load('dCdt.npy', allow_pickle=True)

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
    
    #n_processes = 16
    #seeds_m = np.arange(1,n_processes+1,1)
    #m = []

    #for i in range(n_processes):
    #    m.append(Faas(seed=seeds_m[i]))
    m=Faas(seed=1)
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

    pilot_samples = 200

    # training schedule
    n_train = 500
    n_rounds = 15

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


>>>>>>> c97cae5ab730f186551bf8419c78bcada95f8860

    observation_trace = run_faas_model(true_params)
    observation_summary_statistics = torch.as_tensor(calc_summ(observation_trace))
    snpe_common_args = dict(
        simulator=simulation_wrapper,
        x_o=observation_summary_statistics,
        prior=prior,
        simulation_batch_size=1,
    )


    infer = SnpeC(sample_with_mcmc=False, **snpe_common_args)

<<<<<<< HEAD
    # Run inference.
    num_rounds, num_simulations_per_round = 20, 2000
    posterior = infer(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round, batch_size=2
    )
=======
    print('Sampling!')
    posterior_samples = posterior[0].gen(10000)
    np.save('posterior_samples_summaries.npy', posterior_samples)
>>>>>>> c97cae5ab730f186551bf8419c78bcada95f8860

    samples = posterior.sample(10000)
    np.save('samples.npy', samples)
