import os
import numpy as np
import matplotlib.pyplot as plt
import numpy_indexed as npi
from mouse_fitting_functions import loadMouseWeights
from mouse_plotting_functions import simulateMouse, tolerant_mean, colorFader
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick


SPATH = "Figures/"

plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = (1,1,1,0)
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'     # not available in Colab
plt.rcParams['font.sans-serif'] = 'Helvetica'  # not available in Colab
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 12


def average_subsample(arr, n):
    end =  n * int(np.floor(len(arr)/n))
    return np.mean(arr[:end].reshape(-1, n), 1)


# one must have already fit the mice, either all paths or all names, have to find some warping that allows
# comparison across mice
def plotAverageWeightTrajectories(mouse_names_or_paths, figsize=(15, 10), path=False, save=False):

    if type(mouse_names_or_paths) not in [list, np.ndarray]:
        raise Exception('mouse_names_or_paths must be a list/array of mouse names or paths.')

    n_mice = len(mouse_names_or_paths)

    # array to store the fits of the mice
    fits = np.empty(n_mice, dtype=object)
    # array to store the number of trials of each mouse
    n_trials = np.empty(n_mice)

    # loop over list to populate the fits array and the n_trials array
    for i in range(n_mice):
        fits[i] = loadMouseWeights(mouse_names_or_paths[i], path)
        n_trials[i] = len(fits[i]['w'][0, :])

    # find the smallest number of trials that a mouse has, this will be no. of windows
    n_trials_min = int(np.min(n_trials))

    # create figure and axes
    fig, axs = plt.subplots(2, 2, sharex=True, sharey='row')
    fig.subplots_adjust(hspace=.25, wspace=.25)
    fig.suptitle('Average Weights Across Mice', size=14)
    axs = axs.ravel()
    axs[0].set_title('Left Weight', size=14)
    axs[1].set_title('Right Weight', size=14)
    axs[2].set_title('Left Bias', size=14)
    axs[3].set_title('Right Bias', size=14)

    # arrays to store the weights for the different mice
    right_weights = np.empty((n_mice, n_trials_min))
    left_weights = np.empty((n_mice, n_trials_min))

    # arrays to store the biases for the different mice
    right_bias = np.empty((n_mice, n_trials_min))
    left_bias = np.empty((n_mice, n_trials_min))

    for i in range(n_mice):
        
        if n_trials[i] != n_trials_min:
        
            window_length = int(np.floor(n_trials[i]/n_trials_min))
            remainder = int(n_trials[i] % n_trials_min)
#             print('n_trials', n_trials[i])
#             print('window_length', window_length)
#             print('remainder', remainder)

            if remainder != 0:
        
                right_weights[i, :] = np.mean(fits[i]['w'][0, :-remainder].reshape(-1, window_length), 1)
                right_weights[i, -1] = np.mean(fits[i]['w'][0, -(remainder+window_length):])
                left_weights[i, :] = np.mean(fits[i]['w'][1, :-remainder].reshape(-1, window_length), 1)
                left_weights[i, -1] = np.mean(fits[i]['w'][1, -(remainder+window_length):])
                right_bias[i, :] = np.mean(fits[i]['w'][2, :-remainder].reshape(-1, window_length), 1)
                right_bias[i, -1] = np.mean(fits[i]['w'][2, -(remainder+window_length):])
                left_bias[i, :] = np.mean(fits[i]['w'][3, :-remainder].reshape(-1, window_length), 1)
                left_bias[i, -1] = np.mean(fits[i]['w'][3, -(remainder+window_length):])
            
            if remainder == 0:

                right_weights[i, :] = np.mean(fits[i]['w'][0, :].reshape(-1, window_length), 1)
                left_weights[i, :] = np.mean(fits[i]['w'][1, :].reshape(-1, window_length), 1)
                right_bias[i, :] = np.mean(fits[i]['w'][2, :].reshape(-1, window_length), 1)
                left_bias[i, :] = np.mean(fits[i]['w'][3, :].reshape(-1, window_length), 1)
            
        else:
            right_weights[i, :] = fits[i]['w'][0, :]
            left_weights[i, :] = fits[i]['w'][1, :]
            right_bias[i, :] = fits[i]['w'][2, :]
            left_bias[i, :] = fits[i]['w'][3, :]
        

        axs[0].plot(left_weights[i, :], lw=0.5, color='lightcoral')
        axs[2].plot(left_bias[i, :], lw=0.5, color='lightsalmon')
        axs[1].plot(right_weights[i, :], lw=0.5, color='cornflowerblue')
        axs[3].plot(right_bias[i, :], lw=0.5, color='lightskyblue')

    #sem_right_weights = np.std(right_weights, axis=0)/np.sqrt(n_mice)
    #sem_left_weights = np.std(left_weights, axis=0)/np.sqrt(n_mice)
    #sem_right_bias = np.std(right_bias, axis=0)/np.sqrt(n_mice)
    #sem_left_bias = np.std(left_bias, axis=0)/np.sqrt(n_mice)

    axs[0].plot(np.mean(left_weights, axis=0), alpha=0.8, color='#E50000', lw=2, label='average wl')
    # Plot 95% credible intervals on weights
    #axs[0].fill_between(np.arange(n_trials_min),
    #                 np.mean(left_weights, axis=0) - 1.96 * sem_left_weights,
    #                 np.mean(left_weights, axis=0) + 1.96 * sem_left_weights, color='#E50000', alpha=0.2)

    axs[2].plot(np.mean(left_bias, axis=0), alpha=0.8, color='#FC5A50', lw=2, label='average bl')
    # Plot 95% credible intervals on weights
    #axs[2].fill_between(np.arange(n_trials_min),
    #                    np.mean(left_bias, axis=0) - 1.96 * sem_left_bias,
    #                    np.mean(left_bias, axis=0) + 1.96 * sem_left_bias, color='#FC5A50', alpha=0.2)

    axs[1].plot(np.mean(right_weights, axis=0), alpha=0.8, color='#0000FF', lw=2, label='average wr')
    # Plot 95% credible intervals on weights
    #axs[1].fill_between(np.arange(n_trials_min),
    #                    np.mean(right_weights, axis=0) - 1.96 * sem_right_weights,
    #                    np.mean(right_weights, axis=0) + 1.96 * sem_right_weights, color= '#0000FF', alpha=0.2)

    axs[3].plot(np.mean(right_bias, axis=0), alpha=0.8, color='#069AF3', lw=2, label='average br')
    # Plot 95% credible intervals on weights
    #axs[3].fill_between(np.arange(n_trials_min),
    #                    np.mean(right_bias, axis=0) - 1.96 * sem_right_bias,
    #                    np.mean(right_bias, axis=0) + 1.96 * sem_right_bias, color='#069AF3', alpha=0.2)

    # Further tweaks to make plot nice
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    #axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    #axs.set_ylim(minval, maxval)
    axs[0].set_xlim(0, n_trials_min)
    axs[0].xaxis.set_major_formatter(mtick.PercentFormatter(n_trials_min))
    axs[0].set_ylabel('Parameters', size=14)
    axs[0].legend(loc='upper left')
    
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    #axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    #axs.set_ylim(minval, maxval)
    axs[1].set_xlim(0, n_trials_min)
    axs[1].xaxis.set_major_formatter(mtick.PercentFormatter(n_trials_min))
    axs[1].legend(loc='upper left')
    
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    #axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    #axs.set_ylim(minval, maxval)
    axs[2].set_xlim(0, n_trials_min)
    axs[2].xaxis.set_major_formatter(mtick.PercentFormatter(n_trials_min))
    axs[2].set_xlabel('Percentage of Total Trials', size=14)
    axs[2].set_ylabel('Parameters', size=14)
    axs[2].legend(loc='upper left')
    
    axs[3].spines['right'].set_visible(False)
    axs[3].spines['top'].set_visible(False)
    axs[3].axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    #axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    #axs.set_ylim(minval, maxval)
    axs[3].set_xlim(0, n_trials_min)
    axs[3].xaxis.set_major_formatter(mtick.PercentFormatter(n_trials_min))
    axs[3].set_xlabel('Percentage of Total Trials', size=14)
    axs[3].legend(loc='upper left')

    fig.set_size_inches(figsize)

    if save:
        os.makedirs("{}".format(SPATH), exist_ok=True)
        fig.savefig(SPATH + 'average_weight_trajectories')


def plotAvgWeightQuartiles(mouse_names_or_paths, figsize=(15, 10), path=False, save=False):
    
    np.random.seed(10)
    
    if type(mouse_names_or_paths) not in [list, np.ndarray]:
        raise Exception('mouse_names_or_paths must be a list/array of mouse names or paths.')

    n_mice = len(mouse_names_or_paths)

    # create figure and axes
    fig, axs = plt.subplots(2, 2, sharex=True, sharey='row')
    fig.subplots_adjust(hspace=.25, wspace=.25)
    #fig.suptitle('Average Weights per Quartile Across Mice', size=20)
    axs = axs.ravel()
    axs[0].set_title('Left Weight', size=20)
    axs[1].set_title('Right Weight', size=20)
    axs[2].set_title('Left Bias', size=20)
    axs[3].set_title('Right Bias', size=20)

    # create second figure and axes
    fig2, axs2 = plt.subplots(1, 1)
    axs2.set_title('|Right-Left Bias|', size=20)
    
    # create second figure and axes
    fig3, axs3 = plt.subplots(1, 1)
    axs3.set_title('Sensitivity Bias', size=20)

    # arrays to store the weights for the different mice
    quartile_mean_right_weights = np.empty((n_mice, 4))
    quartile_mean_left_weights = np.empty((n_mice, 4))

    # arrays to store the biases for the different mice
    quartile_mean_right_bias = np.empty((n_mice, 4))
    quartile_mean_left_bias = np.empty((n_mice, 4))
    quartile_mean_right_minus_left_bias = np.empty((n_mice, 4))
    
    quartile_mean_right_minus_left_weight = np.empty((n_mice, 4))
    quartile_mean_weight_avg = np.empty((n_mice, 4))
    
    # arrays to store the average parameters at expert level
    
    expert_mean_right_weights = np.empty((n_mice, 1))
    expert_mean_left_weights = np.empty((n_mice, 1))

    expert_mean_right_bias = np.empty((n_mice, 1))
    expert_mean_left_bias = np.empty((n_mice, 1))
    
    expert_mean_right_minus_left_bias = np.empty((n_mice, 1))

        

    # loop over list to populate the fits array and the n_trials array
    for i in range(n_mice):
        mouse_fit = loadMouseWeights(mouse_names_or_paths[i], path)
        
        corr_tot = (mouse_fit['data']['r'] == (mouse_fit['data']['inputs']['c'] > 0)) * 1

        for j in range(len(corr_tot)):
            if mouse_fit['data']['inputs']['c'][j] == 0:
                corr_tot[j] = (np.random.rand(1) > .5) * 1

        # session info
        max_sessions = len(mouse_fit['data']['dayLength'])
        session_boundaries = np.cumsum(mouse_fit['data']['dayLength'], dtype=int)
        session_boundaries_plt = np.insert(session_boundaries, 0, 0)

        sessions_mean_right_weights = np.empty(max_sessions, dtype=object)
        sessions_mean_left_weights = np.empty(max_sessions, dtype=object)
        sessions_mean_right_bias = np.empty(max_sessions, dtype=object)
        sessions_mean_left_bias = np.empty(max_sessions, dtype=object)
        sessions_mean_right_minus_left_bias = np.empty(max_sessions, dtype=object)
        
        sessions_mean_right_minus_left_weight = np.empty(max_sessions, dtype=object)
        sessions_mean_weight_avg = np.empty(max_sessions, dtype=object)
        
        sessions_av_acc = np.empty(max_sessions, dtype=object)

        for k in range(max_sessions):
            # selecting input data corresponding to session of interest for real mouse
            session_beg = session_boundaries_plt[k]
            session_end = session_boundaries_plt[k + 1]

            sessions_mean_right_weights[k] = np.mean(mouse_fit['w'][0, session_beg:session_end])
            sessions_mean_left_weights[k] = np.mean(mouse_fit['w'][1, session_beg:session_end])
            sessions_mean_right_bias[k] = np.mean(mouse_fit['w'][2, session_beg:session_end])
            sessions_mean_left_bias[k] = np.mean(mouse_fit['w'][3, session_beg:session_end])
            
            sessions_mean_right_minus_left_weight[k] = np.mean((mouse_fit['w'][0, session_beg:session_end] - mouse_fit['w'][1, session_beg:session_end])/ ((abs(mouse_fit['w'][0, session_beg:session_end]) + abs(mouse_fit['w'][1, session_beg:session_end]))/2))
            sessions_mean_weight_avg[k] = np.mean((abs(mouse_fit['w'][0, session_beg:session_end]) + abs(mouse_fit['w'][1, session_beg:session_end]))/2)
            
            sessions_av_acc[k] = np.mean(corr_tot[session_beg:session_end]) 
            
            sessions_mean_right_minus_left_bias[k] = np.mean(np.abs(mouse_fit['w'][2, session_beg:session_end] - mouse_fit['w'][3, session_beg:session_end]))

        # loop over quartiles
        for q in range(4):
            # finding the session indices corresponding to each quartile
            if q <= 2:
                quartile_beg = q * round(max_sessions / 4)
                quartile_end = (q + 1) * round(max_sessions / 4)
            if q == 3:
                quartile_beg = q * round(max_sessions / 4)
                quartile_end = max_sessions

            # arrays to store the weights for the different mice
            quartile_mean_right_weights[i, q] = np.mean(sessions_mean_right_weights[quartile_beg:quartile_end])
            quartile_mean_left_weights[i, q] = np.mean(sessions_mean_left_weights[quartile_beg:quartile_end])

            # arrays to store the biases for the different mice
            quartile_mean_right_bias[i, q] = np.mean(sessions_mean_right_bias[quartile_beg:quartile_end])
            quartile_mean_left_bias[i, q] = np.mean(sessions_mean_left_bias[quartile_beg:quartile_end])
            #quartile_mean_right_minus_left_bias[i, q] = np.mean(np.abs(sessions_mean_right_bias[quartile_beg:quartile_end] - sessions_mean_left_bias[quartile_beg:quartile_end]))#np.mean(sessions_mean_right_minus_left_bias[quartile_beg:quartile_end])
            
            quartile_mean_right_minus_left_weight[i, q] = np.mean(sessions_mean_right_minus_left_weight[quartile_beg:quartile_end])
            quartile_mean_weight_avg[i, q] = np.mean(sessions_mean_weight_avg[quartile_beg:quartile_end])
         
        
        expert_mean_right_weights[i] = np.mean(sessions_mean_right_weights[np.where(sessions_av_acc > 0.75)])
        expert_mean_left_weights[i] = np.mean(sessions_mean_left_weights[np.where(sessions_av_acc > 0.75)])
        expert_mean_right_bias[i] = np.mean(sessions_mean_right_bias[np.where(sessions_av_acc > 0.75)])
        expert_mean_left_bias[i] = np.mean(sessions_mean_left_bias[np.where(sessions_av_acc > 0.75)])
        expert_mean_right_minus_left_bias[i] = np.mean(sessions_mean_right_minus_left_bias[np.where(sessions_av_acc > 0.75)])

        quartile_mean_right_minus_left_bias[i, :] =  np.abs(quartile_mean_right_bias[i, :] - quartile_mean_left_bias[i, :])
        
        
        
        
        if mouse_names_or_paths[i] =='DAP009':
            axs3.plot(quartile_mean_right_minus_left_weight[i, :],quartile_mean_weight_avg[i, :], lw=1, marker = 'o', color='purple', label='DAP009')
            
            axs2.plot(np.arange(4)+1,quartile_mean_right_minus_left_bias[i, :], lw=1, color='purple', label='DAP009')
            axs2.scatter(5,expert_mean_right_minus_left_bias[i], lw=1, color='purple')
            
            axs[0].plot(np.arange(4)+1,quartile_mean_left_weights[i, :], lw=1, color='purple')
            axs[2].plot(np.arange(4)+1,quartile_mean_left_bias[i, :], lw=1, color='purple')
            axs[1].plot(np.arange(4)+1,quartile_mean_right_weights[i, :], lw=1, color='purple')
            axs[3].plot(np.arange(4)+1,quartile_mean_right_bias[i, :], lw=1, color='purple')
            #axs2.plot(quartile_mean_right_minus_left_bias[i, :], lw=0.5, color='green')

            axs[0].scatter(5, expert_mean_left_weights[i], lw=1, color='purple')
            axs[2].scatter(5, expert_mean_left_bias[i], lw=1, color='purple')
            axs[1].scatter(5, expert_mean_right_weights[i], lw=1, color='purple')
            axs[3].scatter(5, expert_mean_right_bias[i], lw=1, color='purple')
            
        elif mouse_names_or_paths[i] =='DAP011':
            axs3.plot(quartile_mean_right_minus_left_weight[i, :],quartile_mean_weight_avg[i, :], lw=1, marker = 'o', color='violet', label = 'DAP011')
            
            axs2.plot(np.arange(4)+1,quartile_mean_right_minus_left_bias[i, :], lw=1, color='violet', label = 'DAP011')
            axs2.scatter(5,expert_mean_right_minus_left_bias[i], lw=1, color='violet')
            
            axs[0].plot(np.arange(4)+1,quartile_mean_left_weights[i, :], lw=1, color='violet')
            axs[2].plot(np.arange(4)+1,quartile_mean_left_bias[i, :], lw=1, color='violet')
            axs[1].plot(np.arange(4)+1,quartile_mean_right_weights[i, :], lw=1, color='violet')
            axs[3].plot(np.arange(4)+1,quartile_mean_right_bias[i, :], lw=1, color='violet')
            #axs2.plot(quartile_mean_right_minus_left_bias[i, :], lw=0.5, color='green')

            axs[0].scatter(5, expert_mean_left_weights[i], lw=1, color='violet')
            axs[2].scatter(5, expert_mean_left_bias[i], lw=1, color='violet')
            axs[1].scatter(5, expert_mean_right_weights[i], lw=1, color='violet')
            axs[3].scatter(5, expert_mean_right_bias[i], lw=1, color='violet')
            
            
        else:
            
            
            axs[0].plot(np.arange(4)+1,quartile_mean_left_weights[i, :], lw=0.5, color='lightcoral')
            axs[2].plot(np.arange(4)+1,quartile_mean_left_bias[i, :], lw=0.5, color='lightsalmon')
            axs[1].plot(np.arange(4)+1,quartile_mean_right_weights[i, :], lw=0.5, color='cornflowerblue')
            axs[3].plot(np.arange(4)+1,quartile_mean_right_bias[i, :], lw=0.5, color='lightskyblue')
            #axs2.plot(quartile_mean_right_minus_left_bias[i, :], lw=0.5, color='green')

            axs[0].scatter(5, expert_mean_left_weights[i], lw=0.5, color='lightcoral', alpha=0.8)
            axs[2].scatter(5, expert_mean_left_bias[i], lw=0.5, color='lightsalmon', alpha=0.8)
            axs[1].scatter(5, expert_mean_right_weights[i], lw=0.5, color='cornflowerblue', alpha=0.8)
            axs[3].scatter(5, expert_mean_right_bias[i], lw=0.5, color='lightskyblue', alpha=0.8)
            
            
            
            axs2.plot(np.arange(4)+1,quartile_mean_right_minus_left_bias[i, :], lw=0.5, color='green')
            axs2.scatter(5,expert_mean_right_minus_left_bias[i], lw=1, color='green', alpha=0.5)
            
            axs3.plot(quartile_mean_right_minus_left_weight[i, :],quartile_mean_weight_avg[i, :], lw=1, marker = 'o', color='gray')
        
        
        

    sem_right_weights = np.std(quartile_mean_right_weights, axis=0)/np.sqrt(n_mice)
    sem_left_weights = np.std(quartile_mean_left_weights, axis=0)/np.sqrt(n_mice)
    sem_right_bias = np.std(quartile_mean_right_bias, axis=0)/np.sqrt(n_mice)
    sem_left_bias = np.std(quartile_mean_left_bias, axis=0)/np.sqrt(n_mice)
    sem_right_minus_left_bias = np.std(quartile_mean_right_minus_left_bias, axis=0)/np.sqrt(n_mice)
    
    sem_expert_right_weights = np.nanstd(expert_mean_right_weights)/np.sqrt(n_mice)
    sem_expert_left_weights = np.nanstd(expert_mean_left_weights)/np.sqrt(n_mice)
    sem_expert_right_bias = np.nanstd(expert_mean_right_bias)/np.sqrt(n_mice)
    sem_expert_left_bias = np.nanstd(expert_mean_left_bias)/np.sqrt(n_mice)
    sem_expert_right_minus_left_bias = np.nanstd(expert_mean_right_minus_left_bias)/np.sqrt(n_mice)
 

    axs[0].errorbar(np.arange(4)+1, np.mean(quartile_mean_left_weights, axis=0), marker='o', yerr= 1.96 * sem_left_weights, alpha=0.8, color='#E50000', lw=3, label='average $w_l$')
    
    axs[0].errorbar(5, np.nanmean(expert_mean_left_weights), marker='o', yerr= 1.96 * sem_expert_left_weights, alpha=0.8, color='#E50000', lw=3)
      
    print(np.mean(expert_mean_left_weights))

    axs[2].errorbar(np.arange(4)+1, np.mean(quartile_mean_left_bias, axis=0), marker='o', yerr= 1.96 * sem_left_bias, alpha=0.8, color='#FC5A50', lw=3, label='average $b_l$')
    
    axs[2].errorbar(5, np.nanmean(expert_mean_left_bias), marker='o', yerr= 1.96 * sem_expert_left_bias, alpha=0.8, color='#FC5A50', lw=3)

    axs[1].errorbar(np.arange(4)+1, np.mean(quartile_mean_right_weights, axis=0), marker='o', yerr= 1.96 * sem_right_weights, alpha=0.8, color='#0000FF', lw=3, label='average $w_r$')
    
    axs[1].errorbar(5, np.nanmean(expert_mean_right_weights), marker='o', yerr= 1.96 * sem_expert_right_weights, alpha=0.8, color='#0000FF', lw=3)

    axs[3].errorbar(np.arange(4)+1, np.mean(quartile_mean_right_bias, axis=0), marker='o', alpha=0.8, yerr= 1.96 * sem_right_bias, color='#069AF3', lw=3, label='average $b_r$')
    
    axs[3].errorbar(5, np.nanmean(expert_mean_right_bias), marker='o', alpha=0.8, yerr= 1.96 * sem_expert_right_bias, color='#069AF3', lw=3)
       
    axs2.errorbar(np.arange(4)+1, np.mean(quartile_mean_right_minus_left_bias, axis=0), marker='o', yerr= 1.96 * sem_right_minus_left_bias, alpha=0.8, color='darkgreen', lw=3, label='average |$b_r$ - $b_l$|')
    
    axs2.errorbar(5, np.nanmean(expert_mean_right_minus_left_bias), marker='o', alpha=1, yerr= 1.96 * sem_expert_right_minus_left_bias, color='darkgreen', lw=3)

    # Further tweaks to make plot nice
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    # axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    # axs.set_ylim(minval, maxval)
    #axs[0].set_xlim(0, n_trials_min)
    axs[0].set_ylabel('Parameters', size=16, labelpad=25)
    axs[0].tick_params(labelsize=14)
    axs[0].legend(loc='upper left', prop={'size': 15})

    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    # axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    # axs.set_ylim(minval, maxval)
    #axs[1].set_xlim(0, n_trials_min)
    axs[1].tick_params(labelsize=14)
    axs[1].legend(loc='upper left', prop={'size': 15})
    
    
    
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    # axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    # axs.set_ylim(minval, maxval)
    #axs[2].set_xlim(0, n_trials_min)
    axs[2].set_xlabel('Quartile #', size=16)
    axs[2].set_ylabel('Parameters', size=16)
    axs[2].set_xticks(np.arange(1, 6, 1))
    axs[2].tick_params(labelsize=14)
    axs[2].legend(loc='upper left', prop={'size': 15})
    

    axs[3].spines['right'].set_visible(False)
    axs[3].spines['top'].set_visible(False)
    axs[3].axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    # axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    # axs.set_ylim(minval, maxval)
    #axs[3].set_xlim(0, n_trials_min)
    axs[3].set_xticks(np.arange(1, 6, 1))
    axs[3].set_xlabel('Quartile #', size=16)
    axs[3].tick_params(labelsize=14)
    axs[3].legend(loc='upper left', prop={'size': 15})
    
    
    

    axs2.spines['right'].set_visible(False)
    axs2.spines['top'].set_visible(False)
    axs2.axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    # axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    # axs.set_ylim(minval, maxval)
    # axs[3].set_xlim(0, n_trials_min)
    axs2.set_xlabel('Quartile #', size=16)
    axs2.set_ylabel('Parameters', size=16, labelpad=25)
    axs2.tick_params(labelsize=14)
    axs2.set_xticks(np.arange(1, 6, 1))
    axs2.legend(loc='upper left', prop={'size': 15})
    
    
    fig.canvas.draw()
    fig2.canvas.draw()
    
    
    labels = [item.get_text() for item in axs[2].get_xticklabels()]
    labels[4] = 'Exp'

    axs[2].set_xticklabels(labels)
    
    
    labels = [item.get_text() for item in axs[3].get_xticklabels()]
    labels[4] = 'Exp'

    axs[3].set_xticklabels(labels)
    
    labels = [item.get_text() for item in axs2.get_xticklabels()]
    labels[4] = 'Exp'

    axs2.set_xticklabels(labels)
    
    
    axs3.spines['right'].set_visible(False)
    axs3.spines['top'].set_visible(False)
    axs3.axvline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    axs3.axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    # axs.set_yticks(np.arange(int(2 * minval), int(2 * maxval) + 1, 1))
    # axs.set_ylim(minval, maxval)
    # axs[3].set_xlim(0, n_trials_min)
    axs3.set_xlabel('Normalised |$w_r$ - $w_l$|', size=16)
    axs3.set_ylabel('mean(|$w_r$|, |$w_l$|)', size=16, labelpad=25)
    axs3.tick_params(labelsize=14)
    #axs3.set_xticks(np.arange(1, 5, 1))
    axs3.legend(loc='upper right', prop={'size': 15})

    fig.set_size_inches(figsize)
    fig2.set_size_inches(figsize)
    fig3.set_size_inches(figsize)

    if save:
        os.makedirs("{}".format(SPATH), exist_ok=True)
        fig.savefig(SPATH + 'quartile_average_weights_over_mice')
        fig2.savefig(SPATH + 'quartile_average_weights_over_mice_bias_difference')




# can only do one of psy_quartiles, rt_quartiles or acc_quartiles - can change so that this is possible
def plotAvgBehavMiceVsModel(mouse_names_or_paths, methods, n_its, figsize=(18, 8), path=False, save=False):

    if type(mouse_names_or_paths) not in [list, np.ndarray]:
        raise Exception('mouse_names_or_paths must be a list/array of mouse names or paths.')

    # useful value
    n_mice = len(mouse_names_or_paths)

    if 'psy_quartiles' in methods or 'rt_quartiles' in methods or 'acc_quartiles' in methods:

        # creating the figure that we will populate in the loops
        fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.5)

        # create inner plots for original data
        inner1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.25, hspace=0.25)
        # set outer title for original data
        out_ax1 = plt.Subplot(fig, outer[0])
        out_ax1.set_title('Original', pad=30, size=18)
        out_ax1.axis('off')
        fig.add_subplot(out_ax1)

        # create inner plots for simulated data
        inner2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.25, hspace=0.25)
        # set outer title for simulated data
        out_ax2 = plt.Subplot(fig, outer[1])
        out_ax2.set_title('Recovered', pad=30, size=18)
        out_ax2.axis('off')
        fig.add_subplot(out_ax2)

        inner_titles = ['Quartile 1', 'Quartile 2', 'Quartile 3', 'Quartile 4']
        inner_axes = np.empty(4, dtype=object)
        inner_axes_sim = np.empty(4, dtype=object)

        for n in range(4):

            # create image plot
            
            if n==0:
                ax = plt.Subplot(fig, inner1[n])
                first_ax = ax
            if n>0:
                ax = plt.Subplot(fig, inner1[n], sharey=first_ax)

            # set inner title
            ax.set_title("{}".format(inner_titles[n]))
            ax.set_xlim([-0.6, 0.6])
            ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
            
            if 'psy_quartiles' in methods or 'acc_quartiles' in methods:
                ax.set_ylim([0, 1])
                ax.set_yticks([0, 0.5, 1])
            
            inner_axes[n] = ax
            fig.add_subplot(ax)

            # create image plot
            ax_sim = plt.Subplot(fig, inner2[n], sharey=first_ax)

            # set inner title
            ax_sim.set_title("{}".format(inner_titles[n]))
            ax_sim.set_xlim([-0.6, 0.6])
            ax_sim.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
            
            if 'psy_quartiles' in methods or 'acc_quartiles' in methods:
                ax_sim.set_ylim([0, 1])
                ax_sim.set_yticks([0, 0.5, 1])
            
            inner_axes_sim[n] = ax_sim
            fig.add_subplot(ax_sim)

        if 'psy_quartiles' in methods:
            mice_resp_quartile = np.empty((n_mice, 4), dtype=object)
            mice_resp_quartile_sim = np.empty((n_mice, 4), dtype=object)

        if 'rt_quartiles' in methods:
            mice_rt_quartile = np.empty((n_mice, 4), dtype=object)
            mice_rt_quartile_sim = np.empty((n_mice, 4), dtype=object)

        if 'acc_quartiles' in methods:
            mice_acc_quartile = np.empty((n_mice, 4), dtype=object)
            mice_acc_quartile_sim = np.empty((n_mice, 4), dtype=object)


    # loop over mice
    for mouse in range(n_mice):

        # load the save_dict for that mouse
        mouse_fit = loadMouseWeights(mouse_names_or_paths[mouse], path)

        # simulation
        resp_sim_iters, rt_sim_iters, contrast_sim_iters, corr_sim_iters = simulateMouse(mouse_fit, n_its, 'random')

        # find corr for all trials of real mouse
        corr_tot = (mouse_fit['data']['r'] == (mouse_fit['data']['inputs']['c'] > 0)) * 1

        for j in range(len(corr_tot)):
            if mouse_fit['data']['inputs']['c'][j] == 0:
                corr_tot[j] = (np.random.rand(1) > .5) * 1

        # session info
        max_sessions = len(mouse_fit['data']['dayLength'])
        session_boundaries = np.cumsum(mouse_fit['data']['dayLength'], dtype=int)
        session_boundaries_plt = np.insert(session_boundaries, 0, 0)

        if 'psy_quartiles' in methods:
            # arrays where session mean/median curves and ci's will be stored for real mouse
            mresp_sessions = np.empty(max_sessions, dtype=object)
            # for simulated mouse
            mresp_sim_sessions = np.empty(max_sessions, dtype=object)

        if 'rt_quartiles' in methods:
            medrt_sessions = np.empty(max_sessions, dtype=object)
            medrt_sim_sessions = np.empty(max_sessions, dtype=object)

        if 'acc_quartiles' in methods:
            macc_sessions = np.empty(max_sessions, dtype=object)
            macc_sim_sessions = np.empty(max_sessions, dtype=object)

        for k in range(max_sessions):
            # selecting input data corresponding to session of interest for real mouse
            session_beg = session_boundaries_plt[k]
            session_end = session_boundaries_plt[k + 1]

            cs = mouse_fit['data']['inputs']['c'][session_beg:session_end]
            contrast_sim_iters_session = contrast_sim_iters[:, session_beg:session_end]
            contrast_sim_concat = contrast_sim_iters_session.flatten()

            if 'psy_quartiles' in methods:
                resp = np.array(mouse_fit['data']['r'][session_beg:session_end], dtype=float)
                resp_sim_iters_session = resp_sim_iters[:, session_beg:session_end]
                resp_sim_concat = resp_sim_iters_session.flatten()
                contrasts, mresp = npi.group_by(cs).mean(resp)
                _, mresp_sim = npi.group_by(contrast_sim_concat).mean(resp_sim_concat)

                mresp_sessions[k] = mresp
                mresp_sim_sessions[k] = mresp_sim

            if 'rt_quartiles' in methods:
                rt = np.array(mouse_fit['data']['T'][session_beg:session_end],
                              dtype=float)  # todo: do these have to be floats?
                rt_sim_iters_session = rt_sim_iters[:, session_beg:session_end]
                rt_sim_concat = rt_sim_iters_session.flatten()
                contrasts, medrt = npi.group_by(cs).median(rt)
                _, medrt_sim = npi.group_by(contrast_sim_concat).median(rt_sim_concat)

                medrt_sessions[k] = medrt
                medrt_sim_sessions[k] = medrt_sim

            if 'acc_quartiles' in methods:
                corr = np.array(corr_tot[session_beg:session_end], dtype=float)
                corr_sim_iters_session = corr_sim_iters[:, session_beg:session_end]
                corr_sim_concat = corr_sim_iters_session.flatten()
                contrasts, macc = npi.group_by(cs).mean(corr)
                _, macc_sim = npi.group_by(contrast_sim_concat).mean(corr_sim_concat)

                macc_sessions[k] = macc
                macc_sim_sessions[k] = macc_sim


        # loop over quartiles
        for q in range(4):
            # finding the session indices corresponding to each quartile
            if q <= 2:
                quartile_beg = q * round(max_sessions / 4)
                quartile_end = (q + 1) * round(max_sessions / 4)
            if q == 3:
                quartile_beg = q * round(max_sessions / 4)
                quartile_end = max_sessions

            if 'psy_quartiles' in methods:
                # extract the mean psy/rt/acc curve for each session in quartile for real mouse
                quartile_session_resp = np.array([x for x in mresp_sessions[quartile_beg:quartile_end]])
                # extract the mean psy/rt/acc curve for each session in quartile for simulated mouse
                quartile_session_sim_resp = np.array([x for x in mresp_sim_sessions[quartile_beg:quartile_end]])
                # store the per quartile mean traces
                mice_resp_quartile[mouse, q] = np.mean(quartile_session_resp, axis=0)
                mice_resp_quartile_sim[mouse, q] = np.mean(quartile_session_sim_resp, axis=0)
                inner_axes[q].plot(contrasts, mice_resp_quartile[mouse, q], color='gray')
                inner_axes_sim[q].plot(contrasts, mice_resp_quartile_sim[mouse, q], linestyle='--', color='gray')

            if 'rt_quartiles' in methods:
                quartile_session_rt = np.array([x for x in medrt_sessions[quartile_beg:quartile_end]])
                quartile_session_sim_rt = np.array([x for x in medrt_sim_sessions[quartile_beg:quartile_end]])
                mice_rt_quartile[mouse, q] = np.mean(quartile_session_rt, axis=0)
                mice_rt_quartile_sim[mouse, q] = np.mean(quartile_session_sim_rt, axis=0)
                inner_axes[q].plot(contrasts, mice_rt_quartile[mouse, q], color='gray')
                inner_axes_sim[q].plot(contrasts, mice_rt_quartile_sim[mouse, q], linestyle='--', color='gray')

            if 'acc_quartiles' in methods:
                quartile_session_acc = np.array([x for x in macc_sessions[quartile_beg:quartile_end]])
                quartile_session_sim_acc = np.array([x for x in macc_sim_sessions[quartile_beg:quartile_end]])
                mice_acc_quartile[mouse, q] = np.mean(quartile_session_acc, axis=0)
                mice_acc_quartile_sim[mouse, q] = np.mean(quartile_session_sim_acc, axis=0)
                inner_axes[q].plot(contrasts, mice_acc_quartile[mouse, q], color='gray')
                inner_axes_sim[q].plot(contrasts, mice_acc_quartile_sim[mouse, q], linestyle='--', color='gray')

    for q in range(4):

        if 'psy_quartiles' in methods:

            mouse_mean_resp = np.mean(mice_resp_quartile[:, q], axis=0)
            ci_mouse_mean_resp = 1.96*np.std(mice_resp_quartile[:, q], axis=0)/np.sqrt(n_mice)

            mouse_mean_resp_sim = np.mean(mice_resp_quartile_sim[:, q], axis=0)
            ci_mouse_mean_resp_sim = 1.96*np.std(mice_resp_quartile_sim[:, q], axis=0)/np.sqrt(n_mice)

            inner_axes[q].errorbar(contrasts, mouse_mean_resp, yerr=ci_mouse_mean_resp, color='k', lw=2.5, markersize=6,
                      capsize=5)
            inner_axes_sim[q].errorbar(contrasts, mouse_mean_resp_sim, yerr=ci_mouse_mean_resp_sim, fmt='--', color='k', lw=2.5, markersize=6,
                      capsize=5)

        if 'rt_quartiles' in methods:

            mouse_mean_rt = np.mean(mice_rt_quartile[:, q], axis=0)
            ci_mouse_mean_rt = 1.96*np.std(mice_rt_quartile[:, q], axis=0)/np.sqrt(n_mice)

            mouse_mean_rt_sim = np.mean(mice_rt_quartile_sim[:, q], axis=0)
            ci_mouse_mean_rt_sim = 1.96*np.std(mice_rt_quartile_sim[:, q], axis=0)/np.sqrt(n_mice)

            inner_axes[q].errorbar(contrasts, mouse_mean_rt, yerr=ci_mouse_mean_rt, color='k', lw=2.5, markersize=6,
                      capsize=5)
            inner_axes_sim[q].errorbar(contrasts, mouse_mean_rt_sim, yerr=ci_mouse_mean_rt_sim, fmt='--', color='k', lw=2.5, markersize=6,
                      capsize=5)

        if 'acc_quartiles' in methods:

            mouse_mean_acc = np.mean(mice_acc_quartile[:, q], axis=0)
            ci_mouse_mean_acc = 1.96*np.std(mice_acc_quartile[:, q], axis=0)/np.sqrt(n_mice)

            mouse_mean_acc_sim = np.mean(mice_acc_quartile_sim[:, q], axis=0)
            ci_mouse_mean_acc_sim = 1.96*np.std(mice_acc_quartile_sim[:, q], axis=0)/np.sqrt(n_mice)

            inner_axes[q].errorbar(contrasts, mouse_mean_acc, yerr=ci_mouse_mean_acc, color='k', lw=2.5, markersize=6,
                      capsize=5)
            inner_axes_sim[q].errorbar(contrasts, mouse_mean_acc_sim, yerr=ci_mouse_mean_acc_sim, fmt='--', color='k', lw=2.5, markersize=6,
                      capsize=5)

    if save:
        if 'psy_quartiles' in methods:
            os.makedirs("{}".format(SPATH), exist_ok=True)
            fig.savefig(SPATH + 'psy_quartiles_over_mice')
            
        if 'rt_quartiles' in methods:
            os.makedirs("{}".format(SPATH), exist_ok=True)
            fig.savefig(SPATH + 'rt_quartiles_over_mice')
        if 'acc_quartiles' in methods:
            fig.savefig('acc_quartiles_over_mice')
            os.makedirs("{}".format(SPATH), exist_ok=True)
            fig.savefig(SPATH + 'acc_quartiles_over_mice')


            
            
            
        
def sessionAlignedAccAcrossMice(mouse_names_or_paths, n_its, figsize=(5, 8), path=False, save=False):
    
    if type(mouse_names_or_paths) not in [list, np.ndarray]:
        raise Exception('mouse_names_or_paths must be a list/array of mouse names or paths.')

    # useful value
    n_mice = len(mouse_names_or_paths)
    n_trials_limit = 50
    
    full_traces = np.empty((n_mice, 4), dtype=object)
    full_traces_sim = np.empty((n_mice, 4), dtype=object)
    
    for mouse in range(n_mice):
        
        # load the save_dict for that mouse
        mouse_fit = loadMouseWeights(mouse_names_or_paths[mouse], path)

        # simulation
        resp_sim_iters, rt_sim_iters, contrast_sim_iters, corr_sim_iters = simulateMouse(mouse_fit, n_its, 'same')

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

       

        # loop over quartiles
        for q in range(4):
            # finding the session indices corresponding to each quartile
            if q <= 2:
                quartile_beg = q * round(max_sessions / 4)
                quartile_end = (q + 1) * round(max_sessions / 4)
            if q == 3:
                quartile_beg = q * round(max_sessions / 4)
                quartile_end = max_sessions

            corr_quartiles = corr_sessions[quartile_beg:quartile_end]
            corr_sim_iters_quartiles = corr_sim_iters_sessions[quartile_beg:quartile_end]
            
            
            
            right_half = np.array([sess[:n_trials_limit] for sess in corr_quartiles], dtype=object)

            # for left half do round which will round up mid-way values e.g. 3.5 -> 4
            left_half = np.array([sess[-n_trials_limit:] for sess in corr_quartiles], dtype=object)


            right_trace = np.array(tolerant_mean(right_half))

            left_trace = np.array(tolerant_mean(left_half))
           
            
            full_traces[mouse, q] = np.convolve(np.concatenate((left_trace, right_trace)), np.ones(17)/17, mode ='valid')

            

            right_half_sim = np.empty(int(n_its*corr_sim_iters_quartiles.shape[0]), dtype=object)
            left_half_sim = np.empty(int(n_its * corr_sim_iters_quartiles.shape[0]), dtype=object)


            count = 0
            for i in range(corr_sim_iters_quartiles.shape[0]):
                for j in range(n_its):

#                     if round(corr_sim_iters_quartiles[i].shape[1] / 2) > n_trials_limit:

                    right_half_sim[count] = corr_sim_iters_quartiles[i][j ,:n_trials_limit]
                    count += 1
#                     else:
#                         right_half_sim[count] = corr_sim_iters_quartiles[i][j ,:int(corr_sim_iters_quartiles[i].shape[1] / 2)]
#                         count += 1


            count = 0
            for i in range(corr_sim_iters_quartiles.shape[0]):
                for j in range(n_its):

#                     if round(corr_sim_iters_quartiles[i].shape[1] / 2) > n_trials_limit:

                    left_half_sim[count] = corr_sim_iters_quartiles[i][j ,-n_trials_limit:]
                    count += 1
#                     else:
#                         left_half_sim[count] = corr_sim_iters_quartiles[i][j ,-int(corr_sim_iters_quartiles[i].shape[1] / 2):]
#                         count += 1

                        
            right_trace_sim = np.array(tolerant_mean(right_half_sim))

            left_trace_sim = np.array(tolerant_mean(left_half_sim))
            
            
            
            full_traces_sim[mouse, q] = np.convolve(np.concatenate((left_trace_sim, right_trace_sim)), np.ones(10)/10, mode='valid')
            
            

    # create figure for each quartile
    fig1, axs1 = plt.subplots(1, 2, sharex=False, sharey=True)
    fig1.subplots_adjust(hspace=.25, wspace=.5)
    fig1.suptitle('Accuracy around Session Boundary', size=14)
    axs1 = axs1.ravel()

    axs1[0].set_ylim(bottom=0.4, top=1)
    axs1[0].set_xlim(-n_trials_limit - 10, n_trials_limit + 10)
    axs1[0].set_xlabel('Trials', size=14)
    axs1[0].set_ylabel('P(correct)', size=14)
    axs1[0].set_title('Original', size=10)
    axs1[0].axvline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    

    axs1[1].set_ylim(bottom=0.4, top=1)
    axs1[1].set_xlim(-n_trials_limit - 10, n_trials_limit + 10)
    axs1[1].set_xlabel('Trials', size=14)
    axs1[1].set_ylabel('P(correct)', size=14)
    axs1[1].set_title('Recovered', size=10)
    axs1[1].axvline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)


    color1 = 'lightskyblue'
    color2 = 'blue'

    for q in range(4):
        
        axs1[0].plot(np.arange(-n_trials_limit+8, n_trials_limit-8), np.mean(full_traces[:, q], axis=0), color=colorFader(color1,color2,q/4))
        
        axs1[1].plot(np.arange(-n_trials_limit+8, n_trials_limit-8), np.mean(full_traces_sim[:, q], axis=0), color=colorFader(color1,color2,q/4), linestyle='--')
            













