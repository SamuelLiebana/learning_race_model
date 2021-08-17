import numpy as np
import os
from datetime import datetime, timedelta
from hyperOpt import hyperOpt
from helper.helperFunctions import trim


def fitBehavData(data, mouse_name=None, session_boundaries=True, N=None, hess_calc='All', save=False, iteration=0):
    """Finds parameters of the model to fit the behavioural data in 'data' (simulated or real).

    Can take in a filepath pointing to data, or the data dict directly. Provide
    a name for the save file, we recommend using the name of the animal whose
    behaviour is being fit (i.e. mouse_name). Specify whether there should be larger
    variances allowed at session boundaries and how many trials of data should be
    recovered. The iteration argument can be used to indicate if more than one fit
    has been performed for a mouse (e.g., if changing initial conditions of
    search). Output is either saved in same folder, or returned directly.
    
    Args:
        data : str or dict, either the filepath to data or a dict containing the data 
        mouse_name : str or None, if provided will be used as the name for the save file
        session_boundaries : bool, indicate whether to allow increased variance at session boundaries
        N : int, number of trials to simulate, if None then just the
            full length of the simulation
        hess_calc : str, passed to hyperOpt(), error bars to calculate ('weights', 'hyper', or 'All')
        save : bool, if True and data is filepath, it saves recovery data as a file in same folder as data;
            if True and data is a dict ,it saves recovery data in folder 'mouse_fits';
            if False, recovery data is returned
        iteration : int, indicate if more than one fit has been performed for a mouse and this will be added
            to the name of the save file
    Returns:
        save_path | (save=True) : str, the name of the folder+file where
            recovery data was saved in the local directory
        save_dict | (save=False) : dict, contains all relevant info
            from the recovery
    """

    # Initialize saved recovery data
    save_dict = {'iteration': iteration}
   
    # Readin simulation input from filepath - the reason we have this is for model recov.
    if type(data) is str:
        if save:
            save_dict['simfile'] = data[:-4]  # use the same name as the datafile for save file (removing extension)
        # todo: this will not work unless data is saved as a dict with the same keys as real mouse data
        readin = np.load(data)

    # this will be the case for real mouse data
    elif isinstance(data, dict):
        # require a mouse name if the input is a dict, otherwise one cannot find the
        # resulting fit files
        if mouse_name is None and save:
            raise Exception('must provide a mouse name for dict input if save is True')
        
        if save:
            # create directory to store fit
            if not os.path.exists('mouse_fits'):
                os.mkdir('mouse_fits')
            
            # root of save path 
            save_dict['simfile'] = os.path.join('mouse_fits', mouse_name)
        
        readin = data
    
    else:
        raise Exception('data must be a string with file name or a dict')
      
    # If number of trials not specified, use all trials of simulation
    if N is None:
        N = len(readin['indices'])
    save_dict['N'] = N
    
    # -------------
    # Fitting
    # -------------

    # Initialization of fitting
    weights = {'wr': 1, 'wl': 1, 'br': 1, 'bl': 1, 'z': 1}
    K = len(weights)  # number of weights to fit
    
    hyper_guess = {
        'sigma': [2**-3, 2**-3, 2**-3, 2**-3, 2**-12],  # arbitrary starting point for the search (z var is approx 0)
        'sigInit': [2 ** 4] * K,  # 2**4 is an arbitrary large value to use for the variance - don't make too large -
                                  # if numerical instability, change to 4
        'sigDay': None,
    }

    optList = ['sigma']  # list of hyperparameters to optimise in hyperOpt
    
    # define dat based on the structure of your input dictionary
    dat = {
        'inputs': {
            'c': readin['c']  # contrast levels
        },
        'r': readin['r'].astype(int),  # responses
        'T': readin['T']  # reaction times
    }

    # Detect whether to include sigDay in optimization
    if session_boundaries:
        
        if 'dayLength' not in readin or readin['dayLength'] is None:
            raise Exception('must provide session/day lengths to enable larger variances at session boundaries')
        
        # 2**-1 is an arbitrary starting point for the search
        hyper_guess['sigDay'] = np.array([2 ** -2, 2 ** -2, 2 ** -2, 2 ** -2, 2 ** -12])
        # add 'sigDay' to hyperparameters to optimise
        optList = ['sigma', 'sigDay']
        dat['dayLength'] = readin['dayLength']

    if N != len(readin['indices']):
        dat = trim(dat, END=N)

    # Run recovery, recording duration of recovery
    START = datetime.now()
    hyp, logEvd, eMode, hess_info = hyperOpt(dat, hyper_guess, weights, optList, hess_calc=hess_calc, showOpt=1) 
    END = datetime.now()
    
    # store the names of the weights
    weight_names = list(weights.keys())
    # reshape learnt parameter vector (eMode) to have params x trials
    w = np.reshape(eMode, (K, N), order='C')
    
    # results to return/save
    save_dict.update({
        'weight_names': weight_names,  # weight/parameter name dictionary
        'data': dat,  # behavioural data used to fit
        'hyp': hyp,  # final hyperparameters
        'logEvd': logEvd,  # final log evidence
        'w': w,  # final parameters
        'hess_info': hess_info,  # confidence intervals for parameters/hyperparameters
        'duration': END - START  # duration of fitting procedure
    })

    # Save or return recovery results is save=True
    if save:
        if N != len(readin['indices']):
            save_path = (save_dict['simfile'] + '_fit' + '_N_' + str(N) + 'i_' + str(iteration) + '.npy')
        else:
            save_path = (save_dict['simfile'] + '_fit' + '_i_' + str(iteration) + '.npy')
        np.save(save_path, save_dict)
        return save_path

    # otherwise directly return the results
    else:
        return save_dict