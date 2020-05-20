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
    prior_max = np.array([12,  10, 12,  10, -3, -4, -4,-4] + [0.05,0.5])
    prior_min = np.array([4,  2, 4, 2, -10, -10, -10, -12] + [-0.05,-0.5])
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

        # true parameters and respective labels
    true_params = np.array([8.886491e+00,  7.924279e+00,
       1.050515e+01,  7.397940e+00, -3.682371e+00, -4.509306e+00, -6.162727e+00,
      -6.585027e+00,  1.100000e-03, -3.900000e-01])
    labels_params = ['KonTN', 'KonTC', 'KonRN', 'KonRC', 'KDTN', 'KDTC', 'KDRN', 'KDRC', 'malpha', 'alpha0']


    observation_trace = run_faas_model(true_params)
    observation_summary_statistics = torch.as_tensor(calc_summ(observation_trace))
    snpe_common_args = dict(
        simulator=simulation_wrapper,
        x_o=observation_summary_statistics,
        prior=prior,
        simulation_batch_size=1,
    )


    infer = SnpeC(sample_with_mcmc=False, **snpe_common_args)


    # Run inference.
    num_rounds, num_simulations_per_round = 2, 30
    posterior = infer(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round, batch_size=2
    )
    print('Sampling!')


    samples = posterior.sample(10000)
    np.save('samples.npy', samples)
