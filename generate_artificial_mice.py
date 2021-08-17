import os
import numpy as np
from datetime import datetime, timedelta
from compute_trial_data import compute_trial_data


def generate_artificial_mice(weight_names=['wr', 'wl', 'br', 'bl', 'z'],
                N=5000,
                hyper={},
                days=None,
                boundary=10.0,
                params=None,
                seed=None,
                savePath=None):
    """Simulates weights, inputs, and choices under the model.
    Args:
        weight_names : list, names of weights to simulate (for us wl, wr, bl, br, z)
        N : int, number of trials to simulate
        hyper : dict, hyperparameters and initial values used to construct the
            prior. Default is none, can include sigma, sigInit, sigDay
        days : list or array, list of the trial indices on which to apply the
            sigDay hyperparameter instead of the sigma
        boundary : float, weights are reflected from this upper boundary
            during simulation (and the lower boundary is 0)
        params : dict, {'dt': float, 'n_steps_per_trial': int, 'sig_i': float, 'sig_o': float}.
        seed : int, random seed to make random simulations reproducible
        savePath : str, if given creates a folder and saves simulation data
            in a file; else data is returned
    Returns:
        save_path | (if savePath) : str, the name of the folder+file where
            simulation data was saved in the local directory
        save_dict | (if no SavePath) : dict, contains all relevant info
            from the simulation
    """

    # get number of weights
    K = len(weight_names)

    # Reproducibility
    np.random.seed(seed)

    # Supply default hyperparameters if necessary
    sigmaDefault = 2**np.random.choice([-4.0, -5.0, -6.0, -7.0, -8.0], size=K)
    sigInitDefault = np.array([4.0] * K)
    sigDayDefault = 2**np.random.choice([1.0, 0.0, -1.0], size=K)

    if 'sigma' not in hyper:
        sigma = sigmaDefault
    elif hyper['sigma'] is None:
        sigma = sigmaDefault
    elif np.isscalar(hyper['sigma']):
        sigma = np.array([hyper['sigma']] * K)
    elif ((type(hyper['sigma']) in [np.ndarray, list]) and
          (len(hyper['sigma']) == K)):
        sigma = hyper['sigma']
    else:
        raise Exception('hyper["sigma"] must be either a scalar or a list or '
                        'array of len K')

    if 'sigInit' not in hyper:
        sigInit = sigInitDefault
    elif hyper['sigInit'] is None:
        sigInit = sigInitDefault
    elif np.isscalar(hyper['sigInit']):
        sigInit = np.array([hyper['sigInit']] * K)
    elif (type(hyper['sigInit']) in [np.ndarray, list]) and (len(hyper['sigInit']) == K):
        sigInit = hyper['sigInit']
    else:
        raise Exception('hyper["sigInit"] must be either a scalar or a list or '
                        'array of len K.')

    if days is None:
        sigDay = None
    elif 'sigDay' not in hyper:
        sigDay = sigDayDefault
    elif hyper['sigDay'] is None:
        sigDay = sigDayDefault
    elif np.isscalar(hyper['sigDay']):
        sigDay = np.array([hyper['sigDay']] * K)
    elif ((type(hyper['sigDay']) in [np.ndarray, list]) and
          (len(hyper['sigDay']) == K)):
        sigDay = hyper['sigDay']
    else:
        raise Exception('hyper["sigDay"] must be either a scalar or a list or '
                        'array of len K.')

    # -------------
    # Simulation
    # -------------

    # Simulate inputs
    c = np.random.choice([-0.5, -0.25, 0, 0.25, 0.5], size=N)

    # Simulate weights
    E = np.zeros((N, K))
    E[0] = np.random.normal(scale=sigInit, size=K)
    E[1:] = np.random.normal(scale=sigma, size=(N - 1, K))
    if sigDay is not None:
        E[np.cumsum(days)] = np.random.normal(scale=sigDay, size=(len(days), K))
    W = np.cumsum(E, axis=0)

    # Impose a ceiling and floor boundary on W
    for i in range(len(W.T)):
        cross = (W[:, i] < 0) | (W[:, i] > boundary)
        while cross.any():
            ind = np.where(cross)[0][0]
            if W[ind, i] < 0:
                W[ind:, i] = -W[ind:, i]
            else:
                W[ind:, i] = 2 * boundary - W[ind:, i]
            cross = (W[:, i] < 0) | (W[:, i] > boundary)

    # default params
    if params is None:
        params = {'dt': .001, 'n_steps_per_trial': 7000, 'sig_i': np.ones(N) * 0.01, 'sig_o': np.ones(N)}

    # weights
    wr = W[:, 0]
    wl = W[:, 1]
    br = W[:, 2]
    bl = W[:, 3]
    z = W[:, 4]

    corr_sim, resp_sim, rt_sim = compute_trial_data(wr, wl, br, bl, z, params['sig_i'],
                                                    params['sig_o'], params['dt'], params['n_steps_per_trial'],
                                                    N, c)

    dat = {
        'inputs': {
            'c': c  # contrast levels
        },
        'r': resp_sim,  # responses
        'T': rt_sim,  # reaction times
        'dayLength' : days
    }

    # Save data
    save_dict = {
        'sigInit': sigInit,
        'weight_names': weight_names,
        'sigDay': sigDay,
        'sigma': sigma,
        'seed': seed,
        'w': W.transpose(),
        'data': dat,
        'K': K,
        'N': N,
        'c' : c,
        'r' : resp_sim,
        'T' : rt_sim,
        'dayLength' : days,
        'hess_info': None,
        'simfile': savePath,
        'indices' : np.arange(len(c))
    }

    # Save & return file path OR return simulation data
    if savePath is not None:
        # Creates unique file name from current datetime
        
        os.makedirs("{}".format('simulated_weights'), exist_ok=True)

        fullSavePath = os.path.join('simulated_weights', savePath)
        np.save(fullSavePath, save_dict)

        return fullSavePath

    else:
        return save_dict


