import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib as mpl
import numpy_indexed as npi
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from compute_trial_data import compute_trial_data


COLORS = {'wr': "#0000FF", 'wl': "#E50000",
          'br': "#069AF3", 'bl': "#FC5A50",
          'z': '#000000'}

ZORDER = {'wr': 1, 'wl': 1,
          'br': 2, 'bl': 2,
          'z': 3}

SPATH = "Figures/"

plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = (1,1,1,0)
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'     # not available in Colab or Jupyter Notebooks
plt.rcParams['font.sans-serif'] = 'Helvetica'  # not available in Colab or Jupyter Notebooks
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 12


def plotWeightTrajectories(mouse_fit, figsize=(15, 5), colors=None, zorder=None, save=False):
    """Plots weight trajectories in a quick and reasonable way.

    Args:
        mouse_fit : save_dict from fitBehavData.
        figsize : size of figure.
        colors : a dict mapping weight names from `weight_dict` to colors.
            Defaults to nice preset values for common weight names.
        zorder : a dict mapping weight names from `weight_dict` to zorder.
            Defaults to nice preset values for common weight names.
        save : bool, whether or not to save the figure as .png.

    Returns:
        fig : The figure, to be modified further if necessary.
    """

    # Some useful values to have around
    K, N = mouse_fit['w'].shape
    maxval = np.max(mouse_fit['w']) * 1.2  # largest magnitude of any weight
    minval = np.min(mouse_fit['w']) - 1
    
    if colors is None:
        colors = COLORS
    if zorder is None:
        zorder = ZORDER

    # Infer (alphabetical) order of weights from dict
    labels = []
    for j in mouse_fit['weight_names']:
        labels += [j]

    # Plot weights and credible intervals
    fig = plt.figure(figsize=figsize)
    axs = fig.add_subplot(1, 1, 1)

    # this if/else is for the simulated weights, which do not have
    # a confidence interval
    if mouse_fit['hess_info'] is not None:
        for i, w in enumerate(labels):
            axs.plot(mouse_fit['w'][i, :], lw=1.5, alpha=0.8, ls='-', c=colors[w],
                     zorder=zorder[w], label=w)
            # Plot 95% credible intervals on weights
            axs.fill_between(np.arange(N),
                             mouse_fit['w'][i, :] - 1.96 * mouse_fit['hess_info']['W_std'][i, :],
                             mouse_fit['w'][i, :] + 1.96 * mouse_fit['hess_info']['W_std'][i, :],
                             facecolor=colors[w], zorder=zorder[w], alpha=0.2)
    else:
        for i, w in enumerate(labels):
            axs.plot(mouse_fit['w'][i, :], lw=1.5, alpha=0.8, ls='-', c=colors[w],
                     zorder=zorder[w], label=w)

    # Plot vertical session lines
    days = mouse_fit['data']['dayLength']
    if days is not None:
        if type(days) not in [list, np.ndarray]:
            raise Exception('days must be a list or array.')
        if np.sum(days) <= N:  # this means day lengths were passed
            days = np.cumsum(days, dtype=int)
        for d in days:
            if d < N:
                axs.axvline(d, c='black', ls='-', lw=0.5, alpha=0.5, zorder=0)

    # Further tweaks to make plot nice
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    axs.set_ylim(minval, maxval)
    axs.set_xlim(0, N)
    axs.set_xlabel('Trial #', size=14)
    axs.set_ylabel('Parameters', size=14)
    axs.legend(loc='upper left')

    if save:
        os.makedirs("{}".format(SPATH), exist_ok=True)
        fig.savefig(SPATH + mouse_fit['simfile']+'_weight_trajectories')

    return fig


def medianCI(data, ci=0.95, p=0.5):
    """Calculates the confidence intervals of the median of a dataset.

    Args:
        data: pandas datafame/series or numpy array.
        ci: confidence level.
        p: percentile' percent, for median it is 0.5.
    Returns:
        a list with two elements, [lowerBound, upperBound].
    """

    if type(data) is pd.Series or type(data) is pd.DataFrame:
        #transfer data into np.array
        data = data.values

    #flat to one dimension array
    data = data.reshape(-1)
    data = np.sort(data)
    N = data.shape[0]
    
    #lowCount, upCount = scipy.stats.binom.interval(ci, N, p, loc=0)
    
    # the guy who implemented this https://github.com/minddrummer/median-confidence-interval
    # used the above, subtracted 1 from lowCount and then applied int()
    
    # I will use this implementation https://www-users.york.ac.uk/~mb55/intro/cicent.htm
    lowCount = N*p - 1.96*np.sqrt(N*p*(1-p))
    upCount = N*p + 1.96*np.sqrt(N*p*(1-p))
    
    lowCount = int(np.ceil(lowCount) - 1)
    upCount = int(np.ceil(upCount) - 1)
    
    if lowCount <0:
        lowCount = 0
    
    if upCount > N-1:
        upCount = N-1
    
    return data[lowCount], data[upCount]


def simulateMouse(mouse_fit, n_its, method='random', params=None):
    """
    Uses the weights in mouse_fit to simulate the LDDM n_its times.

    :param mouse_fit:
    :param params: dict, {'dt': float, 'n_steps_per_trial': int, 'sig_i': float, 'sig_o': float}.
    :param n_its: int, number of times the LDDM should be simulated.
    :return: resp_sim_iters, rt_sim_iters, contrast_sim_iters, corr_sim_iters,: np.arrays with the resulting
             behavioural measures for each iteration.
    """
    
    if method not in ['same', 'random']:
        raise Exception('method must be one of "same" or "random"')

    # useful values
    K, N = mouse_fit['w'].shape

    # default params
    if params is None:
        params = {'dt': .001, 'n_steps_per_trial': 7000, 'sig_i': np.ones(N)*0.01, 'sig_o': np.ones(N)}

    # weights
    wr = mouse_fit['w'][0, :]
    wl = mouse_fit['w'][1, :]
    br = mouse_fit['w'][2, :]
    bl = mouse_fit['w'][3, :]
    z = mouse_fit['w'][4, :]

    # possible contrast levels
    contrasts = np.array([-0.5, -0.25, 0, 0.25, 0.5])  # Contrast levels map onto actual mice data

    # can also try with more contrast levels (though this is not what the actual mouse saw)
    # contrasts = np.linspace(-0.5, 0.5, 10)

    # these arrays contain the responses, rt's and contrasts of each simulation iteration
    resp_sim_iters = np.empty((n_its, N))
    rt_sim_iters = np.empty((n_its, N))
    contrast_sim_iters = np.empty((n_its, N))
    corr_sim_iters = np.empty((n_its, N))

    # run the simulation n_its times
    for i in range(n_its):
        
        if method == 'same':
            cs_sim = mouse_fit['data']['inputs']['c']
            
        #if method == 'balanced': # ask armin whether this is worth implementing
            
        
        elif method == 'random':
            cs_sim = np.random.choice(contrasts, N)
            
        corr_sim, resp_sim, rt_sim = compute_trial_data(wr, wl, br, bl, z, params['sig_i'],
                                                            params['sig_o'], params['dt'], params['n_steps_per_trial'],
                                                            N, cs_sim)
        # save the resulting measures in corresponding arrays
        resp_sim_iters[i, :] = resp_sim
        rt_sim_iters[i, :] = rt_sim
        contrast_sim_iters[i, :] = cs_sim
        corr_sim_iters[i, :] = corr_sim

    return resp_sim_iters, rt_sim_iters, contrast_sim_iters, corr_sim_iters


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# input must be a matrix with windows as rows
def ci_corr(x):
    ci = np.empty((2, x.shape[0]))

    for i in range(x.shape[0]):
        print(i)
        temp = proportion_confint(np.sum(x[i, :]), len(x[i, :]), method='binom_test', alpha=0.05)

        ci[0][i] = temp[0]
        ci[1][i] = temp[1]

    return ci


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


# methods: psy_rt_acc_sessions, psy_rt_acc_quartiles, accuracy_vs_trials, accuracy_vs_sessions, accuracy_vs_quartiles,
# rt_vs_sessions, rt_vs_quartiles, all_psy_rt_acc_quartiles
def plotBehavMouseVsModel(mouse_fit, sim_mouse, methods, figsize=(10, 5), save=False):
    """
    order is not necessarily the same as that provided in methods
    
    """
    
    

    # simulation
    resp_sim_iters, rt_sim_iters, contrast_sim_iters, corr_sim_iters = sim_mouse
    n_its = resp_sim_iters.shape[0]

    # find corr for all trials of real mouse
    corr_tot = (mouse_fit['data']['r'] == (mouse_fit['data']['inputs']['c'] > 0)) * 1

    for j in range(len(corr_tot)):
        if mouse_fit['data']['inputs']['c'][j] == 0:
            corr_tot[j] = (np.random.rand(1) > .5) * 1

    # session info
    max_sessions = len(mouse_fit['data']['dayLength'])
    session_boundaries = np.cumsum(mouse_fit['data']['dayLength'], dtype=int)
    session_boundaries_plt = np.insert(session_boundaries, 0, 0)

    # arrays where session mean/median curves and ci's will be stored for real mouse
    mresp_sessions = np.empty(max_sessions, dtype=object)
    medrt_sessions = np.empty(max_sessions, dtype=object)
    macc_sessions = np.empty(max_sessions, dtype=object)

    session_av_acc = np.empty(max_sessions, dtype=object)
    ci_session_av_acc = np.empty((2, max_sessions))
    session_med_rt = np.empty(max_sessions, dtype=object)
    ci_session_med_rt = np.empty((2, max_sessions))

    # for simulated mouse
    mresp_sim_sessions = np.empty(max_sessions, dtype=object)
    medrt_sim_sessions = np.empty(max_sessions, dtype=object)
    macc_sim_sessions = np.empty(max_sessions, dtype=object)

    session_av_acc_sim = np.empty(max_sessions, dtype=object)
    ci_session_av_acc_sim = np.empty((2, max_sessions))
    session_med_rt_sim = np.empty(max_sessions, dtype=object)
    ci_session_med_rt_sim = np.empty((2, max_sessions))

    for k in range(max_sessions):

        # selecting input data corresponding to session of interest for real mouse
        session_beg = session_boundaries_plt[k]
        session_end = session_boundaries_plt[k + 1]

        # ----------
        # REAL MOUSE
        # ----------

        cs = mouse_fit['data']['inputs']['c'][session_beg:session_end]
        rt = np.array(mouse_fit['data']['T'][session_beg:session_end], dtype=float)  # todo: do these have to be floats?
        resp = np.array(mouse_fit['data']['r'][session_beg:session_end], dtype=float)
        corr = np.array(corr_tot[session_beg:session_end], dtype=float)

        # ---------------
        # SIMULATED MOUSE
        # ---------------

        # selecting input data corresponding to session of interest for simulated mouse
        resp_sim_iters_session = resp_sim_iters[:, session_beg:session_end]
        rt_sim_iters_session = rt_sim_iters[:, session_beg:session_end]
        contrast_sim_iters_session = contrast_sim_iters[:, session_beg:session_end]
        corr_sim_iters_session = corr_sim_iters[:, session_beg:session_end]

        # we will average over all the iterations, so we flatten the above arrays
        resp_sim_concat = resp_sim_iters_session.flatten()
        rt_sim_concat = rt_sim_iters_session.flatten()
        contrast_sim_concat = contrast_sim_iters_session.flatten()
        corr_sim_concat = corr_sim_iters_session.flatten()

        if 'accuracy_vs_sessions' in methods or 'accuracy_vs_quartiles' in methods:

            # ----------
            # REAL MOUSE
            # ----------

            # find average accuracy in this session
            session_av_acc[k] = np.mean(corr)

            # find confidence interval for this mean
            temp = proportion_confint(np.sum(corr), len(corr), method='binom_test', alpha=0.05)
            lower = session_av_acc[k] - temp[0]
            upper = temp[1] - session_av_acc[k]
            ci_session_av_acc[0][k] = lower
            ci_session_av_acc[1][k] = upper

            # ---------------
            # SIMULATED MOUSE
            # ---------------

            # find average accuracy in this simulated session
            session_av_acc_sim[k] = np.mean(corr_sim_concat)

            # find confidence interval for this mean
            temp = proportion_confint(np.sum(corr_sim_concat), len(corr_sim_concat), method='binom_test', alpha=0.05)
            lower = session_av_acc_sim[k] - temp[0]
            upper = temp[1] - session_av_acc_sim[k]
            ci_session_av_acc_sim[0][k] = lower
            ci_session_av_acc_sim[1][k] = upper

        if 'rt_vs_sessions' in methods or 'rt_vs_quartiles' in methods:

            # ----------
            # REAL MOUSE
            # ----------

            # find median rt in this session
            session_med_rt[k] = np.median(rt)

            # find confidence interval for this median
            temp = medianCI(rt, 0.95, 0.5)
            ci_session_med_rt[0][k] = session_med_rt[k] - temp[0]
            ci_session_med_rt[1][k] = temp[1] - session_med_rt[k]

            # ---------------
            # SIMULATED MOUSE
            # ---------------

            # find median rt in this simulated session
            session_med_rt_sim[k] = np.median(rt_sim_concat)

            # find confidence interval for this median
            temp = medianCI(rt_sim_concat, 0.95, 0.5)
            ci_session_med_rt_sim[0][k] = session_med_rt_sim[k] - temp[0]
            ci_session_med_rt_sim[1][k] = temp[1] - session_med_rt_sim[k]

        # --------------------------------------------------
        # Psychometric curves / RT curves / ACC vs cs curves
        # --------------------------------------------------

        if 'psy_rt_acc_sessions' in methods or 'psy_rt_acc_quartiles' in methods or 'all_psy_rt_acc_quartiles' in methods:

            # ----------
            # REAL MOUSE
            # ----------

            # find mean psych, median rt, and mean acc curves for real mouse
            contrasts, mresp = npi.group_by(cs).mean(resp)
            _, medrt = npi.group_by(cs).median(rt)
            _, macc = npi.group_by(cs).mean(corr)

            # ----------------------------------------------
            # find confidence intervals for mean psych curve
            # -----------------------------------------------

            # number of right choices and total number of trials for each
            # contrast level (used to find binom ci's)
            resp_split_successes = npi.group_by(cs).sum(resp)
            split_total = npi.group_by(cs).count

            # calculate confidence intervals for psychometric curve and store in ci_resp for each cs value
            ci_resp = np.empty((2, len(split_total)))
            for j in range(len(split_total)):
                
                try:
                    temp = proportion_confint(resp_split_successes[1][j], split_total[j], method='binom_test', alpha=0.05)
                except:
                    print(resp_split_successes[1][j])
                    print(split_total[j])
                    temp = proportion_confint(resp_split_successes[1][j], split_total[j]+1, method='binom_test', alpha=0.05)
                    
                lower = mresp[j] - temp[0]
                upper = temp[1] - mresp[j]
                ci_resp[0][j] = lower
                ci_resp[1][j] = upper

            # ----------------------------------------
            # find confidence intervals for median RT
            # ----------------------------------------

            # split the rt's by cs value
            split_rt = npi.group_by(cs).split(rt)

            # apply median confidence interval func and place ci's into 2 x num_cs_groups array
            ci_rt = np.empty((2, len(split_total)))
            for j in range(len(split_total)):
                temp = medianCI(split_rt[j], 0.95, 0.5)
                ci_rt[0][j] = medrt[j] - temp[0]
                ci_rt[1][j] = temp[1] - medrt[j]

            # ----------------------------------------
            # find confidence intervals for mean acc
            # ----------------------------------------

            # number of correct choices and total number of trials for each
            # contrast level (used to find binom ci's)
            acc_split_successes = npi.group_by(cs).sum(corr)

            # calculate confidence intervals for acc curve and store in ci_acc for each cs value
            ci_acc = np.empty((2, len(split_total)))
            for j in range(len(split_total)):
                temp = proportion_confint(acc_split_successes[1][j], split_total[j], method='binom_test', alpha=0.05)
                lower = macc[j] - temp[0]
                upper = temp[1] - macc[j]
                ci_acc[0][j] = lower
                ci_acc[1][j] = upper

            # store the mean/median psych/rt curves and ci's for each session
            mresp_sessions[k] = [mresp, ci_resp]
            medrt_sessions[k] = [medrt, ci_rt]
            macc_sessions[k] = [macc, ci_acc]

            # ---------------
            # SIMULATED MOUSE
            # ---------------

            # find mean psych and median rt curve for simulated mouse
            _, mresp_sim = npi.group_by(contrast_sim_concat).mean(resp_sim_concat)
            _, medrt_sim = npi.group_by(contrast_sim_concat).median(rt_sim_concat)
            _, macc_sim = npi.group_by(contrast_sim_concat).mean(corr_sim_concat)

            # ----------------------------------------------
            # find confidence intervals for mean psych curve
            # -----------------------------------------------

            resp_split_successes_sim = npi.group_by(contrast_sim_concat).sum(resp_sim_concat)
            split_total_sim = npi.group_by(contrast_sim_concat).count
            
            ci_resp_sim = np.empty((2, len(split_total_sim)))
            for j in range(len(split_total_sim)):
                temp = proportion_confint(resp_split_successes_sim[1][j], split_total_sim[j], method='binom_test', alpha=0.05)
                lower = mresp_sim[j] - temp[0]
                upper = temp[1] - mresp_sim[j]
                ci_resp_sim[0][j] = lower
                ci_resp_sim[1][j] = upper

            # ----------------------------------------
            # find confidence intervals for median RT
            # ----------------------------------------

            split_rt_sim = npi.group_by(contrast_sim_concat).split(rt_sim_concat)
            ci_rt_sim = np.empty((2, len(split_total_sim)))
            for j in range(len(split_rt_sim)):
                temp = medianCI(split_rt_sim[j], 0.95, 0.5)
                ci_rt_sim[0][j] = medrt_sim[j] - temp[0]
                ci_rt_sim[1][j] = temp[1] - medrt_sim[j]

            # ----------------------------------------
            # find confidence intervals for mean acc
            # ----------------------------------------

            acc_split_successes_sim = npi.group_by(contrast_sim_concat).sum(corr_sim_concat)
            ci_acc_sim = np.empty((2, len(split_total_sim)))
            for j in range(len(split_total)):
                temp = proportion_confint(acc_split_successes_sim[1][j], split_total_sim[j], method='binom_test', alpha=0.05)
                lower = macc_sim[j] - temp[0]
                upper = temp[1] - macc_sim[j]
                ci_acc_sim[0][j] = lower
                ci_acc_sim[1][j] = upper

            # store the mean/median psych/rt curves and ci's for each simulated session
            mresp_sim_sessions[k] = [mresp_sim, ci_resp_sim]
            medrt_sim_sessions[k] = [medrt_sim, ci_rt_sim]
            macc_sim_sessions[k] = [macc_sim, ci_acc_sim]

            if 'psy_rt_acc_sessions' in methods:

                # create figure and axes
                fig1, axs1 = plt.subplots(1, 3, sharex=True, sharey=False)
                fig1.subplots_adjust(hspace=.25, wspace=.25)
                fig1.suptitle('Session '+str(k+1), size=14)
                axs1 = axs1.ravel()

                # properties of psych curve plot
                axs1[0].set_ylim(bottom=0, top=1)
                axs1[0].set_xlim(-0.6, 0.6)
                axs1[0].set_xlabel('Contrast', size=14)
                axs1[0].set_ylabel('P(right)', size=14)
                axs1[0].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))

                # properties of RT plot
                axs1[1].set_ylim(bottom=0, top=3)
                axs1[1].set_xlim(-0.6, 0.6)
                axs1[1].set_xlabel('Contrast', size=14)
                axs1[1].set_ylabel('Median Response Time', size=14)
                axs1[1].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))

                # properties of acc plot
                axs1[2].set_ylim(bottom=0, top=1)
                axs1[2].set_xlim(-0.6, 0.6)
                axs1[2].set_xlabel('Contrast', size=14)
                axs1[2].set_ylabel('P(correct)', size=14)
                axs1[2].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))

                # original
                axs1[0].errorbar(contrasts, mresp, yerr=ci_resp, color='k', markersize=6, capsize=5, label='original')
                axs1[1].errorbar(contrasts, medrt, yerr=ci_rt, color='k', markersize=6, capsize=5, label='original')
                axs1[2].errorbar(contrasts, macc, yerr=ci_acc, color='k', markersize=6, capsize=5, label='original')

                # simulation
                axs1[0].errorbar(contrasts, mresp_sim, yerr=ci_resp_sim, color='r', fmt='--', markersize=6, capsize=5,
                                 label='recovered')
                axs1[1].errorbar(contrasts, medrt_sim, yerr=ci_rt_sim, color='r', fmt='--', markersize=6, capsize=5,
                                 label='recovered')
                axs1[2].errorbar(contrasts, macc_sim, yerr=ci_acc_sim, color='r', fmt='--', markersize=6, capsize=5,
                                 label='recovered')

                # add legend
                axs1[0].legend()
                axs1[1].legend()
                axs1[2].legend()

                # set figure size
                fig1.set_size_inches(figsize)

                # save figure
                if save:
                    os.makedirs("{}".format(SPATH), exist_ok=True)
                    fig1.savefig(SPATH+mouse_fit['simfile']+'psy_rt_acc_session_'+ str(k+1))

    if 'psy_rt_acc_quartiles' in methods or 'all_psy_rt_acc_quartiles' in methods or 'accuracy_vs_quartiles' in methods or 'rt_vs_quartiles' in methods:

        if 'accuracy_vs_quartiles' in methods:

            quartile_av_acc = np.empty(4, dtype=object)
            ci_quartile_av_acc = np.empty(4, dtype=object)

            quartile_av_acc_sim = np.empty(4, dtype=object)
            ci_quartile_av_acc_sim = np.empty(4, dtype=object)

        if 'rt_vs_quartiles' in methods:

            quartile_med_rt = np.empty(4, dtype=object)
            ci_quartile_med_rt = np.empty(4, dtype=object)

            quartile_med_rt_sim = np.empty(4, dtype=object)
            ci_quartile_med_rt_sim = np.empty(4, dtype=object)

        if 'all_psy_rt_acc_quartiles' in methods:

            mean_quartile_session_resp = np.empty(4, dtype=object)
            mean_quartile_session_rt = np.empty(4, dtype=object)
            mean_quartile_session_acc = np.empty(4, dtype=object)

            mean_quartile_session_sim_resp = np.empty(4, dtype=object)
            mean_quartile_session_sim_rt = np.empty(4, dtype=object)
            mean_quartile_session_sim_acc = np.empty(4, dtype=object)

            store_sem_quartile_mean_resp = np.empty(4, dtype=object)
            store_sem_quartile_sim_mean_resp = np.empty(4, dtype=object)

            store_sem_quartile_mean_rt = np.empty(4, dtype=object)
            store_sem_quartile_sim_mean_rt = np.empty(4, dtype=object)

            store_sem_quartile_mean_acc = np.empty(4, dtype=object)
            store_sem_quartile_sim_mean_acc = np.empty(4, dtype=object)

        # loop over quartiles
        for q in range(4):
            # finding the session indices corresponding to each quartile
            if q <= 2:
                quartile_beg = q * round(max_sessions/4)
                quartile_end = (q + 1) * round(max_sessions/4)
            if q == 3:
                quartile_beg = q * round(max_sessions/4)
                quartile_end = max_sessions

            if 'psy_rt_acc_quartiles' in methods or 'all_psy_rt_acc_quartiles' in methods:

                # extract the mean psy/rt/acc curve for each session in quartile for real mouse
                quartile_session_resp = np.array([x[0] for x in mresp_sessions[quartile_beg:quartile_end]])
                quartile_session_rt = np.array([x[0] for x in medrt_sessions[quartile_beg:quartile_end]])
                quartile_session_acc = np.array([x[0] for x in macc_sessions[quartile_beg:quartile_end]])

                # extract the ci's for each session in quartile for real mouse todo: ask Armin if this is necessary
                #quartile_session_ci_resp = np.array([x[1] for x in mresp_sessions[quartile_beg:quartile_end]])
                #quartile_session_ci_rt = np.array([x[1] for x in medrt_sessions[quartile_beg:quartile_end]])
                #quartile_session_ci_acc = np.array([x[1] for x in macc_sessions[quartile_beg:quartile_end]])

                # extract the mean psy/rt/acc curve for each session in quartile for simulated mouse
                quartile_session_sim_resp = np.array([x[0] for x in mresp_sim_sessions[quartile_beg:quartile_end]])
                quartile_session_sim_rt = np.array([x[0] for x in medrt_sim_sessions[quartile_beg:quartile_end]])
                quartile_session_sim_acc = np.array([x[0] for x in macc_sim_sessions[quartile_beg:quartile_end]])

                # extract the ci's for each session in quartile for real mouse
                #quartile_session_ci_sim_resp = np.array([x[1] for x in mresp_sim_sessions[quartile_beg:quartile_end]])
                #quartile_session_ci_sim_rt = np.array([x[1] for x in medrt_sim_sessions[quartile_beg:quartile_end]])
                #quartile_session_ci_sim_acc = np.array([x[1] for x in macc_sim_sessions[quartile_beg:quartile_end]])

                if 'psy_rt_acc_quartiles' in methods:

                    # create figure for each quartile
                    fig2, axs2 = plt.subplots(2, 1, sharex=True, sharey=True)
                    fig2.subplots_adjust(hspace=.25, wspace=.25)
                    fig2.suptitle('Quartile '+str(q+1), size=14)
                    axs2 = axs2.ravel()

                    axs2[0].set_ylim(bottom=0, top=1)
                    axs2[0].set_xlim(-0.6, 0.6)
                    axs2[0].set_xlabel('Contrast', size=14)
                    axs2[0].set_ylabel('P(right)', size=14)
                    axs2[0].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
                    axs2[0].set_title('Original', size=14)

                    axs2[1].set_ylim(bottom=0, top=1)
                    axs2[1].set_xlim(-0.6, 0.6)
                    axs2[1].set_xlabel('Contrast', size=14)
                    axs2[1].set_ylabel('P(right)', size=14)
                    axs2[1].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
                    axs2[1].set_title('Recovered', size=14)

                    fig3, axs3 = plt.subplots(2, 1, sharex=True, sharey=True)
                    fig3.subplots_adjust(hspace=.5, wspace=.25)
                    fig3.suptitle('Quartile ' + str(q+1), size=14)
                    axs3 = axs3.ravel()

                    axs3[0].set_ylim(bottom=0, top=5)
                    axs3[0].set_xlabel('Contrast', size=14)
                    axs3[0].set_ylabel('Median Response Time', size=14)
                    axs3[0].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
                    axs3[0].set_title('Original', size=14)  # :)

                    axs3[1].set_ylim(bottom=0, top=5)
                    axs3[1].set_xlabel('Contrast', size=14)
                    axs3[1].set_ylabel('Median Response Time', size=14)
                    axs3[1].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
                    axs3[1].set_title('Recovered', size=14)

                    fig4, axs4 = plt.subplots(2, 1, sharex=True, sharey=True)
                    fig4.subplots_adjust(hspace=.5, wspace=.25)
                    fig4.suptitle('Quartile ' + str(q+1), size=14)
                    axs4 = axs4.ravel()

                    axs4[0].set_ylim(bottom=0, top=1)
                    axs4[0].set_xlabel('Contrast', size=14)
                    axs4[0].set_ylabel('P(correct)', size=14)
                    axs4[0].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
                    axs4[0].set_title('Original', size=14)  # :)

                    axs4[1].set_ylim(bottom=0, top=1)
                    axs4[1].set_xlabel('Contrast', size=14)
                    axs4[1].set_ylabel('P(correct)', size=14)
                    axs4[1].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
                    axs4[1].set_title('Recovered', size=14)

                    # parameters of colour gradient
                    color1='lightskyblue' 
                    color2='blue' 
                    num_colors= quartile_end - quartile_beg -1

                    # plot the session by session psy/rt/acc curves in a lighter colour
                    for s in range(quartile_end - quartile_beg):

                        # session by session without error bars
                        axs2[0].plot(contrasts, quartile_session_resp[s], color=colorFader(color1,color2,s/num_colors))
                        axs2[1].plot(contrasts, quartile_session_sim_resp[s], linestyle='dashed', color=colorFader(color1,color2,s/num_colors))

                        axs3[0].plot(contrasts, quartile_session_rt[s], color=colorFader(color1,color2,s/num_colors))
                        axs3[1].plot(contrasts, quartile_session_sim_rt[s], linestyle='dashed', color=colorFader(color1,color2,s/num_colors))

                        axs4[0].plot(contrasts, quartile_session_acc[s], color=colorFader(color1,color2,s/num_colors))
                        axs4[1].plot(contrasts, quartile_session_sim_acc[s], linestyle='dashed', color=colorFader(color1,color2,s/num_colors))

                        # session by session with error bars
                        # axs2[0].errorbar(contrasts, quartile_session_resp[s], yerr = quartile_session_ci_resp[s], color = 'lightgray', markersize=6, capsize=5)
                        # axs2[1].errorbar(contrasts, quartile_session_sim_resp[s], yerr = quartile_session_ci_sim_resp[s], color = 'lightpink', markersize=6, capsize=5)

                        # axs3[0].errorbar(contrasts, quartile_session_rt[s], yerr = quartile_session_ci_rt[s], color = 'lightgray', markersize=6, capsize=5)
                        # axs3[1].errorbar(contrasts, quartile_session_sim_rt[s], yerr = quartile_session_ci_sim_rt[s], color = 'lightpink', markersize=6, capsize=5)

                        # axs4[0].errorbar(contrasts, quartile_session_acc[s], yerr = quartile_session_ci_acc[s], color = 'lightgray', markersize=6, capsize=5)
                        # axs4[1].errorbar(contrasts, quartile_session_sim_acc[s], yerr = quartile_session_ci_sim_acc[s], color = 'lightpink', markersize=6, capsize=5)

                # calculate the ci's of the quartile averages
                sem_quartile_mean_resp = np.std(quartile_session_resp, axis=0) / np.sqrt(quartile_end - quartile_beg)
                sem_quartile_sim_mean_resp = np.std(quartile_session_sim_resp, axis=0) / np.sqrt(
                    quartile_end - quartile_beg)

                sem_quartile_mean_rt = np.std(quartile_session_rt, axis=0) / np.sqrt(quartile_end - quartile_beg)
                sem_quartile_sim_mean_rt = np.std(quartile_session_sim_rt, axis=0) / np.sqrt(quartile_end - quartile_beg)

                sem_quartile_mean_acc = np.std(quartile_session_acc, axis=0) / np.sqrt(quartile_end - quartile_beg)
                sem_quartile_sim_mean_acc = np.std(quartile_session_sim_acc, axis=0) / np.sqrt(quartile_end - quartile_beg)

                if 'psy_rt_acc_quartiles' in methods:

                    # plot the quartile averages in a darker colour superimposed on the session-by-session traces with errorbars
                    axs2[0].errorbar(contrasts, np.mean(quartile_session_resp, axis=0), yerr=1.96 * sem_quartile_mean_resp, color='k', linewidth= 3, markersize=6, capsize=5)
                    axs2[1].errorbar(contrasts, np.mean(quartile_session_sim_resp, axis=0),
                                     yerr=1.96 * sem_quartile_sim_mean_resp, color='k', fmt='--', linewidth= 3, markersize=6, capsize=5)

                    axs3[0].errorbar(contrasts, np.mean(quartile_session_rt, axis=0), yerr=1.96 * sem_quartile_mean_rt,
                                     color='k', linewidth= 3, markersize=6, capsize=5)
                    axs3[1].errorbar(contrasts, np.mean(quartile_session_sim_rt, axis=0), yerr=1.96 * sem_quartile_sim_mean_rt,
                                     color='k', fmt='--', linewidth= 3, markersize=6, capsize=5)

                    axs4[0].errorbar(contrasts, np.mean(quartile_session_acc, axis=0), yerr=1.96 * sem_quartile_mean_acc,
                                     color='k', linewidth= 3, markersize=6, capsize=5)
                    axs4[1].errorbar(contrasts, np.mean(quartile_session_sim_acc, axis=0), yerr=1.96 * sem_quartile_sim_mean_acc,
                                     color='k', fmt='--', linewidth= 3, markersize=6, capsize=5)

                    # set the figure sizes
                    fig2.set_size_inches(figsize)
                    fig3.set_size_inches(figsize)
                    fig4.set_size_inches(figsize)

                    # save figure
                    if save:
                        os.makedirs("{}".format(SPATH), exist_ok=True)
                        fig2.savefig(SPATH+mouse_fit['simfile']+'psy_quartile_'+str(q+1))
                        fig3.savefig(SPATH+mouse_fit['simfile']+'rt_quartile_'+str(q+1))
                        fig4.savefig(SPATH+mouse_fit['simfile']+'acc_quartile_'+str(q+1))

                if 'all_psy_rt_acc_quartiles' in methods:

                    # store the per quartile mean traces
                    mean_quartile_session_resp[q] = np.mean(quartile_session_resp, axis=0)
                    mean_quartile_session_rt[q] = np.mean(quartile_session_rt, axis=0)
                    mean_quartile_session_acc[q] = np.mean(quartile_session_acc, axis=0)

                    mean_quartile_session_sim_resp[q] = np.mean(quartile_session_sim_resp, axis=0)
                    mean_quartile_session_sim_rt[q] = np.mean(quartile_session_sim_rt, axis=0)
                    mean_quartile_session_sim_acc[q] = np.mean(quartile_session_sim_acc, axis=0)

                    # store the sem's of the mean traces per quartile
                    store_sem_quartile_mean_resp[q] = sem_quartile_mean_resp
                    store_sem_quartile_sim_mean_resp[q] = sem_quartile_sim_mean_resp

                    store_sem_quartile_mean_rt[q] = sem_quartile_mean_rt
                    store_sem_quartile_sim_mean_rt[q] = sem_quartile_sim_mean_rt

                    store_sem_quartile_mean_acc[q] = sem_quartile_mean_acc
                    store_sem_quartile_sim_mean_acc[q] = sem_quartile_sim_mean_acc

            if 'accuracy_vs_quartiles' in methods:

                # ----------
                # REAL MOUSE
                # ----------

                # quartile average
                quartile_av_acc[q] = np.mean(session_av_acc[quartile_beg:quartile_end])
                # ci of quartile average
                ci_quartile_av_acc[q] = 1.96*np.std(session_av_acc[quartile_beg:quartile_end])/np.sqrt(quartile_end-quartile_beg)

                # ---------------
                # SIMULATED MOUSE
                # ---------------

                # quartile average
                quartile_av_acc_sim[q] = np.mean(session_av_acc_sim[quartile_beg:quartile_end])
                # ci of quartile average
                ci_quartile_av_acc_sim[q] = 1.96*np.std(session_av_acc_sim[quartile_beg:quartile_end]) / np.sqrt(
                    quartile_end - quartile_beg)

            if 'rt_vs_quartiles' in methods:

                # ----------
                # REAL MOUSE
                # ----------

                # quartile average
                quartile_med_rt[q] = np.mean(session_med_rt[quartile_beg:quartile_end])
                # ci of quartile average
                ci_quartile_med_rt[q] = 1.96 * np.std(session_med_rt[quartile_beg:quartile_end]) / np.sqrt(
                    quartile_end - quartile_beg)

                # ---------------
                # SIMULATED MOUSE
                # ---------------

                # quartile average
                quartile_med_rt_sim[q] = np.mean(session_med_rt_sim[quartile_beg:quartile_end])
                # ci of quartile average
                ci_quartile_med_rt_sim[q] = 1.96 * np.std(session_med_rt_sim[quartile_beg:quartile_end]) / np.sqrt(
                    quartile_end - quartile_beg)

        if 'all_psy_rt_acc_quartiles' in methods:

            # create figure for each of psy/rt/acc
            fig5, axs5 = plt.subplots(1, 2, sharex=True, sharey=True)
            fig5.subplots_adjust(hspace=.25, wspace=.25)
            fig5.suptitle('Psychometric Curves over Quartiles', size=14)
            axs5 = axs5.ravel()

            axs5[0].set_ylim(bottom=0, top=1)
            axs5[0].set_xlim(-0.6, 0.6)
            axs5[0].set_xlabel('Contrast', size=14)
            axs5[0].set_ylabel('P(right)', size=14)
            axs5[0].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
            axs5[0].set_title('Original', size=14)

            axs5[1].set_ylim(bottom=0, top=1)
            axs5[1].set_xlim(-0.6, 0.6)
            axs5[1].set_xlabel('Contrast', size=14)
            axs5[1].set_ylabel('P(right)', size=14)
            axs5[1].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
            axs5[1].set_title('Recovered', size=14)

            fig6, axs6 = plt.subplots(1, 2, sharex=True, sharey=True)
            fig6.subplots_adjust(hspace=.25, wspace=.25)
            fig6.suptitle('Median Response Time vs. Contrast over Quartiles', size=14)
            axs6 = axs6.ravel()

            axs6[0].set_ylim(bottom=0, top=3)
            axs6[0].set_xlabel('Contrast', size=14)
            axs6[0].set_ylabel('Median Response Time', size=14)
            axs6[0].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
            axs6[0].set_title('Original', size=14)  # :)

            axs6[1].set_ylim(bottom=0, top=3)
            axs6[1].set_xlabel('Contrast', size=14)
            axs6[1].set_ylabel('Median Response Time', size=14)
            axs6[1].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
            axs6[1].set_title('Recovered', size=14)

            fig7, axs7 = plt.subplots(1, 2, sharex=True, sharey=True)
            fig7.subplots_adjust(hspace=.25, wspace=.25)
            fig7.suptitle('Accuracy vs. Contrast Over Quartiles', size=14)
            axs7 = axs7.ravel()

            axs7[0].set_ylim(bottom=0, top=1)
            axs7[0].set_xlabel('Contrast', size=14)
            axs7[0].set_ylabel('P(correct)', size=14)
            axs7[0].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
            axs7[0].set_title('Original', size=14)  # :)

            axs7[1].set_ylim(bottom=0, top=1)
            axs7[1].set_xlabel('Contrast', size=14)
            axs7[1].set_ylabel('P(correct)', size=14)
            axs7[1].set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
            axs7[1].set_title('Recovered', size=14)

            # parameters of colour gradient
            color1 = 'lightskyblue'  
            color2 = 'blue'  
            num_colors = 4

            for num in range(num_colors):

                # plot the quartile averages in a colour progression
                axs5[0].errorbar(contrasts, mean_quartile_session_resp[num], yerr=1.96 * store_sem_quartile_mean_resp[num],
                                 color=colorFader(color1,color2,num/num_colors), linewidth=3, markersize=6, capsize=5)
                axs5[1].errorbar(contrasts, mean_quartile_session_sim_resp[num],
                                 yerr=1.96 * store_sem_quartile_sim_mean_resp[num], color=colorFader(color1,color2,num/num_colors), fmt='--', linewidth=3, markersize=6, capsize=5)

                axs6[0].errorbar(contrasts, mean_quartile_session_rt[num], yerr=1.96 * store_sem_quartile_mean_rt[num],
                                 color=colorFader(color1,color2,num/num_colors), linewidth=3, markersize=6, capsize=5)
                axs6[1].errorbar(contrasts, mean_quartile_session_sim_rt[num], yerr=1.96 * store_sem_quartile_sim_mean_rt[num],
                                 color=colorFader(color1,color2,num/num_colors), fmt='--', linewidth=3, markersize=6, capsize=5)

                axs7[0].errorbar(contrasts, mean_quartile_session_acc[num], yerr=1.96 * store_sem_quartile_mean_acc[num],
                                 color=colorFader(color1,color2,num/num_colors), linewidth=3, markersize=6, capsize=5)
                axs7[1].errorbar(contrasts, mean_quartile_session_sim_acc[num], yerr=1.96 * store_sem_quartile_sim_mean_acc[num],
                                 color=colorFader(color1,color2,num/num_colors), fmt='--', linewidth=3, markersize=6, capsize=5)

            # set the figure sizes
            fig5.set_size_inches(figsize)
            fig6.set_size_inches(figsize)
            fig7.set_size_inches(figsize)

            # save figure
            if save:
                os.makedirs("{}".format(SPATH), exist_ok=True)
                fig5.savefig(SPATH+mouse_fit['simfile'] + 'all_psy_quartiles')
                fig6.savefig(SPATH+mouse_fit['simfile'] + 'all_rt_quartiles')
                fig7.savefig(SPATH+mouse_fit['simfile'] + 'all_acc_quartiles')

        if 'accuracy_vs_quartiles' in methods:

            fig8 = plt.figure()
            axs8 = fig8.add_subplot(1, 1, 1)
            axs8.set_ylim(bottom=0.2, top=1)
            axs8.set_xlim(left=0.5, right=4.5)
            axs8.set_xlabel('Quartile #', size=14)
            axs8.set_ylabel('P(correct)', size=14)
            axs8.set_xticks(np.arange(1, 5))
            axs8.set_title('Accuracy vs. Quartiles', size=14)
            axs8.errorbar(np.arange(1, 5), quartile_av_acc, yerr=ci_quartile_av_acc, color='k', markersize=6,
                          capsize=5, label='original')
            axs8.errorbar(np.arange(1, 5), quartile_av_acc_sim, yerr=ci_quartile_av_acc_sim, color='r', fmt='--',
                          markersize=6, capsize=5, label='recovered')
            axs8.legend()

            # save figure
            if save:
                os.makedirs("{}".format(SPATH), exist_ok=True)
                fig8.savefig(SPATH+mouse_fit['simfile'] + 'accuracy_vs_quartiles')

        if 'rt_vs_quartiles' in methods:

            fig9 = plt.figure()
            axs9 = fig9.add_subplot(1, 1, 1)
            axs9.set_ylim(bottom=0, top=5)
            axs9.set_xlim(left=0.5, right=4.5)
            axs9.set_xlabel('Quartile #', size=14)
            axs9.set_ylabel('Median Response Time', size=14)
            axs9.set_xticks(np.arange(1, 5))
            #axs9.set_title('Median Response Time vs. Quartiles', size=14)
            axs9.errorbar(np.arange(1, 5), quartile_med_rt, yerr=ci_quartile_med_rt, color='k', markersize=6,
                          capsize=5, label='original')
            axs9.errorbar(np.arange(1, 5), quartile_med_rt_sim, yerr=ci_quartile_med_rt_sim, color='r', fmt='--',
                          markersize=6, capsize=5, label='recovered')
            axs9.legend()

            # save figure
            if save:
                os.makedirs("{}".format(SPATH), exist_ok=True)
                fig9.savefig(SPATH+mouse_fit['simfile'] + 'rt_vs_quartiles')

    if 'accuracy_vs_sessions' in methods:

        fig10 = plt.figure()
        axs10 = fig10.add_subplot(1, 1, 1)
        axs10.set_ylim(bottom=0.2, top=1)
        axs10.set_xlim(left=0.5, right=max_sessions+1)
        axs10.set_xlabel('Session #', size=14)
        axs10.set_xticks(np.arange(1, max_sessions+1, 2))
        axs10.set_ylabel('P(correct)', size=14)
        #axs10.set_title('Accuracy vs. Sessions', size=14)
#         axs10.errorbar(np.arange(1, max_sessions+1), session_av_acc, yerr=ci_session_av_acc, color='k', markersize=6,
#                       capsize=5, label='original')
#         axs10.errorbar(np.arange(1, max_sessions+1), session_av_acc_sim, yerr=ci_session_av_acc_sim, color='r', fmt='--',
#                       markersize=6, capsize=5, label='recovered')

        axs10.plot(np.arange(max_sessions)+1, np.array(session_av_acc, dtype=float), label='original', c='k')
        axs10.fill_between(np.arange(max_sessions)+1, np.array(session_av_acc-ci_session_av_acc[0], dtype=float), np.array(session_av_acc+ci_session_av_acc[1], dtype=float), facecolor='k', alpha=0.1)
        
        axs10.plot(np.arange(max_sessions)+1, np.array(session_av_acc_sim, dtype=float), label='recovered', c='r', linestyle='dashed')
        axs10.fill_between(np.arange(max_sessions)+1, np.array(session_av_acc_sim-ci_session_av_acc_sim[0], dtype=float), np.array(session_av_acc_sim+ci_session_av_acc_sim[1], dtype=float), facecolor='r', alpha=0.1)


        axs10.legend()

        # save figure
        if save:
            os.makedirs("{}".format(SPATH), exist_ok=True)
            fig10.savefig(SPATH+mouse_fit['simfile'] + 'accuracy_vs_sessions')

    if 'rt_vs_sessions' in methods:

        fig11 = plt.figure()
        axs11 = fig11.add_subplot(1, 1, 1)
        axs11.set_ylim(bottom=0, top=3)
        axs11.set_xlim(left=0.5, right=max_sessions+1)
        axs11.set_xlabel('Session #', size=14)
        axs11.set_xticks(np.arange(1, max_sessions+1, 2))
        axs11.set_ylabel('Median Response Time', size=14)
        #axs11.set_title('Median Response Time vs. Sessions', size=14)
        
#         axs11.errorbar(np.arange(1, max_sessions+1), session_med_rt, yerr=ci_session_med_rt, color='k', markersize=6,
#                       capsize=5, label='original')
#         axs11.errorbar(np.arange(1, max_sessions+1), session_med_rt_sim, yerr=ci_session_med_rt_sim, color='r', fmt='--',
#                       markersize=6, capsize=5, label='recovered')
        
        
        
        axs11.plot(np.arange(max_sessions)+1, np.array(session_med_rt, dtype=float), label='original', c='k')
        axs11.fill_between(np.arange(max_sessions)+1, np.array(session_med_rt-ci_session_med_rt[0], dtype=float), np.array(session_med_rt+ci_session_med_rt[1], dtype=float), facecolor='k', alpha=0.1)
        
        axs11.plot(np.arange(max_sessions)+1, np.array(session_med_rt_sim, dtype=float), label='recovered', c='r', linestyle='dashed')
        axs11.fill_between(np.arange(max_sessions)+1, np.array(session_med_rt_sim-ci_session_med_rt_sim[0], dtype=float), np.array(session_med_rt_sim+ci_session_med_rt_sim[1], dtype=float), facecolor='r', alpha=0.1)
        
        
        axs11.legend()

        # save figure
        if save:
            os.makedirs("{}".format(SPATH), exist_ok=True)
            fig11.savefig(SPATH+mouse_fit['simfile'] + 'rt_vs_sessions')

    if 'accuracy_vs_trials' in methods: # might remove this

        window_length = 2

        # ----------
        # REAL MOUSE
        # ----------

        acc = np.convolve(corr_tot, np.ones(window_length) / window_length, mode='valid')  # mean trace
        ci = ci_corr(rolling_window(corr_tot, window_length))  # confidence intervals

        # ---------------
        # SIMULATED MOUSE
        # ---------------

        # array which will contain the moving averages of the corr for each iteration
        acc_sim = np.empty(n_its, dtype=object)

        # this list will be populated with the windows used to
        # calculated the moving average, but the nice thing is that
        # we collect all the data from all the iterations into each window
        # to get a better estimate of the average accuracy in that window
        corr_list_tot_sim = [[] for i in range(len(acc))]  # n.b. len(acc) is tot # windows
        for i in range(n_its):
            # divide each corr trace from each iteration into windows
            windows = rolling_window(corr_sim_iters[i, :], window_length)
            # append the windows to the corresponding element of corr_list_tot_sim
            for j in range(len(acc)):
                corr_list_tot_sim[j].extend(windows[j].tolist())
            # to later find the mean trace we will calculate the moving average for each corr iteration
            acc_sim[i] = np.convolve(corr_sim_iters[i, :], np.ones(window_length) / window_length, mode='valid')

        # convert list to numpy array to then apply ci_corr
        corr_list_tot_sim = np.array(corr_list_tot_sim)

        # now that we have the data from all iterations split into
        # windows we can apply the ci_corr function
        ci_sim = ci_corr(corr_list_tot_sim)

        # we can now plot the resulting curves and error bars
        fig12 = plt.figure()
        axs12 = fig12.add_subplot(1, 1, 1)
        axs12.set_title('Accuracy vs. Trials', size=14)
        axs12.set_xlabel('Trial', size=14)
        axs12.set_ylabel('Accuracy', size=14)
        axs12.set_ylim([0.4, 1])
        axs12.set_xticks(np.arange(1, len(acc)+1))
        axs12.set_xlim(left=0.5, right=len(acc)+0.5)
        axs12.plot(np.arange(1, len(acc)+1), acc, label='original')
        axs12.fill_between(np.arange(1, len(acc)+1), ci[0][:], ci[1][:], alpha=0.2)
        axs12.plot(np.arange(1, len(acc)+1), np.mean(acc_sim, axis=0), 'r--', label='recovered')
        axs12.fill_between(np.arange(1, len(acc)+1), ci_sim[0][:], ci_sim[1][:], color='r', alpha=0.2)
        axs12.legend()

        if save:
            os.makedirs("{}".format(SPATH), exist_ok=True)
            fig12.savefig(SPATH+mouse_fit['simfile'] + 'accuracy_vs_trials')

    if 'rt_vs_trials' in methods:

        # all rt's of real mouse
        rt_tot = mouse_fit['data']['T']

        # rt's averaged per iteration
        rt_tot_sim = np.mean(rt_sim_iters, axis=0)
        # sem*1.96
        ci_r_tot_sim = 1.96*np.std(rt_sim_iters, axis=0)/n_its

        # we can now plot the resulting curves and error bars
        fig13 = plt.figure()
        axs13 = fig13.add_subplot(1, 1, 1)
        axs13.set_title('Median Response Time vs. Trials', size=14)
        axs13.set_xlabel('Trial', size=14)
        axs13.set_ylabel('Median Response Time', size=14)
        axs13.set_ylim([0, 5])
        axs13.set_xticks(np.arange(1, len(rt_tot) + 1))
        axs13.set_xlim(left=0.5, right=len(rt_tot) + 0.5)
        axs13.plot(np.arange(1, len(rt_tot) + 1), rt_tot, label='original')
        axs13.plot(np.arange(1, len(rt_tot) + 1), rt_tot_sim, 'r--', label='recovered')
        axs13.fill_between(np.arange(1, len(rt_tot) + 1), rt_tot_sim - ci_r_tot_sim, rt_tot_sim + ci_r_tot_sim, color='r', alpha=0.2)
        axs13.legend()

        if save:
            os.makedirs("{}".format(SPATH), exist_ok=True)
            fig13.savefig(SPATH+mouse_fit['simfile'] + 'rt_vs_trials')


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1)#, arr.std(axis=-1)


def sessionAlignedAcc(mouse_fit, sim_mouse, figsize=(5, 10), save=False):
    # simulation
    resp_sim_iters, rt_sim_iters, contrast_sim_iters, corr_sim_iters = sim_mouse
    n_its = resp_sim_iters.shape[0]

    # find corr for all trials of real mouse
    corr_tot = (mouse_fit['data']['r'] == (mouse_fit['data']['inputs']['c'] > 0)) * 1

    for j in range(len(corr_tot)):
        if mouse_fit['data']['inputs']['c'][j] == 0:
            corr_tot[j] = (np.random.rand(1) > .5) * 1

    # session info
    max_sessions = len(mouse_fit['data']['dayLength'])
    session_boundaries = np.cumsum(mouse_fit['data']['dayLength'], dtype=int)
    session_boundaries_plt = np.insert(session_boundaries, 0, 0)

    corr_sessions = np.empty(max_sessions, dtype=object)
    corr_sim_iters_sessions = np.empty(max_sessions, dtype=object)

    for k in range(max_sessions):

        # selecting input data corresponding to session of interest for real mouse
        session_beg = session_boundaries_plt[k]
        session_end = session_boundaries_plt[k + 1]

        # ----------
        # REAL MOUSE
        # ----------

        corr_sessions[k] = np.array(corr_tot[session_beg:session_end], dtype=float)

        # ---------------
        # SIMULATED MOUSE
        # ---------------

        # selecting input data corresponding to session of interest for simulated mouse
        corr_sim_iters_sessions[k] = corr_sim_iters[:, session_beg:session_end]
    
    corr_quartiles = np.empty(4, dtype=object)
    corr_sim_iters_quartiles = np.empty(4, dtype=object)
    n_trials_median = np.empty(4)

    # loop over quartiles
    for q in range(4):
        # finding the session indices corresponding to each quartile
        if q <= 2:
            quartile_beg = q * round(max_sessions / 4)
            quartile_end = (q + 1) * round(max_sessions / 4)
        if q == 3:
            quartile_beg = q * round(max_sessions / 4)
            quartile_end = max_sessions

        corr_quartiles[q] = corr_sessions[quartile_beg:quartile_end]
        corr_sim_iters_quartiles[q] = corr_sim_iters_sessions[quartile_beg:quartile_end]

        n_trials = np.array([len(session) for session in corr_quartiles[q]])
        n_trials_median[q] = np.median(n_trials)
        

    n_trials_limit = int(np.min(n_trials_median)/2)  # rounds down - that's fine

    # create figure for each quartile
    fig1, axs1 = plt.subplots(2, 1, sharex=False, sharey=True)
    fig1.subplots_adjust(hspace=.5, wspace=.25)
    fig1.suptitle('Accuracy around Session Boundary', size=14)
    axs1 = axs1.ravel()

    axs1[0].set_ylim(bottom=0.4, top=1)
    axs1[0].set_xlim(-n_trials_limit - 10, n_trials_limit + 10)
    axs1[0].set_xlabel('Trials', size=14)
    axs1[0].set_ylabel('P(correct)', size=14)
    axs1[0].set_title('Original', size=14)
    axs1[0].axvline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)

    axs1[1].set_ylim(bottom=0.4, top=1)
    axs1[1].set_xlim(-n_trials_limit - 10, n_trials_limit + 10)
    axs1[1].set_xlabel('Trials', size=14)
    axs1[1].set_ylabel('P(correct)', size=14)
    axs1[1].set_title('Recovered', size=14)
    axs1[1].axvline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    


    color1 = 'lightskyblue'
    color2 = 'blue'

    # loop over quartiles once again
    for q in range(4):
        
        # consider removing the len(sess)/2 requiremient
        right_half = np.array([sess[:n_trials_limit] for sess in corr_quartiles[q]], dtype=object)

        # for left half do round which will round up mid-way values e.g. 3.5 -> 4
        left_half = np.array([sess[-n_trials_limit:] for sess in corr_quartiles[q]], dtype=object)
        
        
        right_trace = np.array(tolerant_mean(right_half))

        left_trace = np.array(tolerant_mean(left_half))

        axs1[0].plot(np.arange(-n_trials_limit+8, n_trials_limit-8), np.convolve(np.concatenate((left_trace, right_trace)), np.ones(17)/17, mode ='valid'), color=colorFader(color1,color2,q/4))
        
        
        

        right_half_sim = np.empty(int(n_its*corr_sim_iters_quartiles[q].shape[0]), dtype=object)
        left_half_sim = np.empty(int(n_its * corr_sim_iters_quartiles[q].shape[0]), dtype=object)
        

        count = 0
        for i in range(corr_sim_iters_quartiles[q].shape[0]):
            for j in range(n_its):
                
                #if round(corr_sim_iters_quartiles[q][i].shape[1] / 2) > n_trials_limit:

                right_half_sim[count] = corr_sim_iters_quartiles[q][i][j ,:n_trials_limit]
                count += 1
#                 else:
#                     right_half_sim[count] = corr_sim_iters_quartiles[q][i][j ,:int(corr_sim_iters_quartiles[q][i].shape[1] / 2)]
#                     count += 1


        count = 0
        for i in range(corr_sim_iters_quartiles[q].shape[0]):
            for j in range(n_its):
                
                

                left_half_sim[count] = corr_sim_iters_quartiles[q][i][j ,-n_trials_limit:]
                count += 1
#                 else:
#                     left_half_sim[count] = corr_sim_iters_quartiles[q][i][j ,-int(corr_sim_iters_quartiles[q][i].shape[1] / 2):]
#                     count += 1

        right_trace_sim = np.array(tolerant_mean(right_half_sim))

        left_trace_sim = np.array(tolerant_mean(left_half_sim))

        axs1[1].plot(np.arange(-n_trials_limit+8, n_trials_limit-8), np.convolve(np.concatenate((left_trace_sim, right_trace_sim)), np.ones(17)/17, mode='valid'), color=colorFader(color1,color2,q/4), linestyle='--')
        

#         axs1[1].plot(np.arange(-n_trials_limit, n_trials_limit), np.concatenate((left_trace_sim, right_trace_sim)), color=colorFader(color1,color2,q/4), linestyle='--')

        fig1.set_size_inches(figsize)
    
    if save:
        os.makedirs("{}".format(SPATH), exist_ok=True)
        fig1.savefig(SPATH + mouse_fit['simfile']+'_acc_session_boundary')











