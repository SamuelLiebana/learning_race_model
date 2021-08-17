import numpy as np


def compute_trial_data(wr, wl, br, bl, z, sig_i, sig_o, dt, n_steps_per_trial, n_trials, c):
    """
    Function to compute simulated behavioural data from a collection of parameters
    of the race model.

    Args:
        wr : array, Right response contrast-to-drift-rate proportionality constant
        wl : array, Left response contrast-to-drift-rate proportionality constant
        br : array, right bias input to integrator
        bl: array, left bias input to integrator
        z : array, Response threshold
        n_input : float, input noise variance
        n : float, output noise variance
        dt : float, simulation timestep
        n_steps_per_trial : int, max timesteps to roll out on each trial
        n_trials : int, number of trials to simulate
        c : array, vector of contrast levels to simulate. Positive = right
    Returns:
        corr : array, n_trials-element binary vector indicating correct trials
        resp : array, n_trials-element binary vector indicating response direction (0=left)
        rt : n_trials-element vector indicating reaction time
    """

    # ---
    # Reshaping
    # ---

    # ensure these are column vectors
    z = z.reshape(-1, 1)
    sig_i = sig_i.reshape(-1, 1)
    sig_o = sig_o.reshape(-1, 1)
   
    # ensure these are row vectors
    #wr = np.reshape(wr, (1, len(wr)))
    #wl = np.reshape(wl, (1, len(wl)))

    # ---
    # Eta will be the noise deriving from the input (w_k**2 sigma_i**2)
    # ---

    # tiling wr and wl for every timestep so that we can multiply
    # by an independent gaussian sample to model the noise
    wr_for_eta = np.reshape(wr, (1, len(wr)))
    wr_for_eta = np.tile(wr_for_eta.transpose(), (1, n_steps_per_trial))  # had a .transpose here before
    
    wl_for_eta = np.reshape(wl, (1, len(wl)))
    wl_for_eta = np.tile(wl_for_eta.transpose(), (1, n_steps_per_trial))

    Eta_r = np.sqrt(dt)*np.random.normal(0, 1, (n_trials, n_steps_per_trial)) * np.multiply(sig_i, wr_for_eta)
    Eta_l = np.sqrt(dt)*np.random.normal(0, 1, (n_trials, n_steps_per_trial)) * np.multiply(sig_i, wl_for_eta)

    # ---
    # output noise
    # ---

    # we do this separately to avoid any correlations
    Out_r = np.multiply(sig_o, np.random.normal(0, 1, (n_trials, n_steps_per_trial)))*np.sqrt(dt)
    Out_l = np.multiply(sig_o, np.random.normal(0, 1, (n_trials, n_steps_per_trial)))*np.sqrt(dt)

    # ---
    # drift terms
    # ---

    A = wr * np.maximum(np.zeros(n_trials), c)
    A = np.reshape(A, (1, len(A)))
    A_r = np.tile(A.transpose(), (1, n_steps_per_trial)) * dt  # A_r is matrix of shape (n_trials, n_steps_per_trial)  * dt

    A2 = (-1) * wl * np.minimum(np.zeros(n_trials), c)
    A2 = np.reshape(A2, (1, len(A2)))
    A_l = np.tile(A2.transpose(), (1, n_steps_per_trial)) * dt
    
    br = np.reshape(br, (1, len(br)))
    Bias_r = np.tile(br.transpose(), (1, n_steps_per_trial)) * dt  # similarly for the bias terms
    
    bl = np.reshape(bl, (1, len(bl)))
    Bias_l = np.tile(bl.transpose(), (1, n_steps_per_trial)) * dt

    # ---
    # Accumulator Values and Hitting Times
    # ---

    # accumulators values = drift Integral + Brownian motion + bias + output noise:
    int_r = np.cumsum(A_r + Bias_r + Eta_r + Out_r, axis=1)
    int_l = np.cumsum(A_l + Bias_l + Eta_l + Out_l, axis=1)    # row-wise cumsum: per trial

    # counts time points until the integrators cross threshold with cumulative
    # product of row-wise of boolean that indicates whether int was > threshold
    # and then a sum to see how many trials passed until there was a 0
    hitting_r = np.sum(np.cumprod((int_r > z) == 0, axis=1), axis=1)
    hitting_l = np.sum(np.cumprod((int_l > z) == 0, axis=1), axis=1)

    # this sets the accumulators values to 1 for all those above the
    # threshold (probability mass in top stays in top for rest of trial)
    # doesn't really matter for our fits though (we just care about the hitting
    # time and which integrator hit first)
    for i in range(n_trials):
        int_l[i, hitting_l[i]:n_steps_per_trial] = z[i]
        int_r[i, hitting_r[i]:n_steps_per_trial] = z[i]

    # turns timesteps into a reaction time with *0.01
    hitting_l = hitting_l*dt
    hitting_r = hitting_r*dt

    # ---
    # Data to Save
    # ---

    # sets the RT to the hitting time of the fastest integrator
    rt = np.minimum(hitting_l, hitting_r)

    # which response is predicted based on which one hit threshold sooner: 0 = Left, 1 = Right
    resp = (((hitting_l == hitting_r)*(np.random.rand(len(hitting_r)) > .5)) * 1) \
        + (~(hitting_l == hitting_r) * (hitting_r < hitting_l) * 1)

    # is the response above correct? compare to ground truth
    corr = (resp == (c > 0)) * 1

    # we have to adjust for the 0-contrast trials, that is,
    # if c==0  randomly correct as ground truth doesnt have preference
    for trial in range(len(corr)):
        if c[trial] == 0:
            corr[trial] = (np.random.rand(1) > .5) * 1

    return corr, resp, rt
