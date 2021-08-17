import os
import numpy as np
import pandas as pd
from itertools import groupby
from fitting_function import fitBehavData


def readTable(mouse_name):
    """Loads the data corresponding to mouse name.

    Args:
        mouse_name : string, name of mouse whose data you want to load.
    Returns:
        data : dict, contains choices, RT's, contrast levels, and session lengths
        for the mouse in question.
    """

    # check argument type
    if type(mouse_name) is not str:
        raise Exception('readTable_error: mouse_name must be a string')

    # read in all data from csv file - this should be changed to file in server
    table_df = pd.read_csv('./mouse_data/LearningGrating2AFC_trialInfo_behav.csv')
    # extract the mouse names of each trial from the expref
    mouse_names = np.array([table_df['expRef'][i][-6:] for i in range(len(table_df['expRef']))])
    # find the indices of the df where the mouse name corresponds to the current mouse_name
    mouse_indexes = np.where(mouse_names == mouse_name)[0]
    # raise exception if mouse_name is not in dataset
    if not np.any(mouse_indexes):
        raise Exception('readTable_error: mouse_name is invalid, no mouse with that name in dataset')
    # store the contrast levels for each trial of that mouse (-ve -> left, +ve -> right)
    c = table_df['contrastRight'][mouse_indexes] - table_df['contrastLeft'][mouse_indexes]
    # define reaction times as the time between choice completion and stimulus onset
    T = table_df['choiceCompleteTime'][mouse_indexes] - table_df['stimulusOnsetTime'][mouse_indexes]
    # store the responses of each trial ('Left', 'Right', and 'NoGo')
    r = table_df['choice'][mouse_indexes].copy()
    # store the entire expref for all the trials of the mouse in question
    sessions = table_df['expRef'][mouse_indexes]

    # remove repeatNumber >1 and trialNumber <= 5
    c.drop(table_df['repeatNumber'][mouse_indexes].index[
                (table_df['repeatNumber'][mouse_indexes] > 1) | (table_df['trialNumber'][mouse_indexes] <= 5)],
            inplace=True)
    T.drop(table_df['repeatNumber'][mouse_indexes].index[
               (table_df['repeatNumber'][mouse_indexes] > 1) | (table_df['trialNumber'][mouse_indexes] <= 5)],
           inplace=True)
    sessions.drop(table_df['repeatNumber'][mouse_indexes].index[
                      (table_df['repeatNumber'][mouse_indexes] > 1) | (table_df['trialNumber'][mouse_indexes] <= 5)],
                  inplace=True)
    r.drop(table_df['repeatNumber'][mouse_indexes].index[
               (table_df['repeatNumber'][mouse_indexes] > 1) | (table_df['trialNumber'][mouse_indexes] <= 5)],
           inplace=True)

    # remove NoGo trials
    c.drop(r.index[r == 'NoGo'], inplace=True)
    T.drop(r.index[r == 'NoGo'], inplace=True)
    sessions.drop(r.index[r == 'NoGo'], inplace=True)
    r.drop(r.index[r == 'NoGo'], inplace=True)

    # remove reaction time > 5
    c.drop(T.index[T > 5], inplace=True)
    r.drop(T.index[T > 5], inplace=True)
    sessions.drop(T.index[T > 5], inplace=True)
    T.drop(T.index[T > 5], inplace=True)

    r[r == 'Left'] = 0.0  # map 'Left' to 0
    r[r == 'Right'] = 1.0  # map 'Right' to 1

    # store cs, T, and r as numpy arrays in a dictionary
    data = {'r': r.to_numpy(), 'T': T.to_numpy(), 'c': c.to_numpy()}
    
    # ----------------------------------------------------------------------------
    # This is an optional part of the readTable function which would allow you 
    # to balance the number of trials for each contrast level - not generally used
    # -----------------------------------------------------------------------------
    
    #maskminus012 = np.random.choice(np.where(inputs['cs'] == -0.12)[0], len(inputs['cs'][inputs['cs']==0.12]), replace=False)

    #mask025 = np.random.choice(np.where(inputs['cs'] == 0.25)[0], len(inputs['cs'][inputs['cs']==0.12]), replace=False)

    #mask05 = np.random.choice(np.where(inputs['cs'] == 0.5)[0], len(inputs['cs'][inputs['cs']==0.25]), replace=False)

    #mask0 = np.random.choice(np.where(inputs['cs'] == 0)[0], len(inputs['cs'][inputs['cs']==0.12]), replace=False)

    #maskminus025 = np.random.choice(np.where(inputs['cs'] == -0.25)[0], len(inputs['cs'][inputs['cs']==0.12]), replace=False)

    #maskminus05 = np.random.choice(np.where(inputs['cs'] == -0.5)[0], len(inputs['cs'][inputs['cs']==0.25]), replace=False)


    # mask = np.hstack([np.where(inputs['cs'] <= 0)[0], mask012, mask025, mask05])

    #mask = np.hstack([np.where((inputs['cs'] != 0.5) & (inputs['cs'] != -0.5))[0], mask05, maskminus05])
    #mask = np.sort(mask)

    #inputs['r'] = inputs['r'][mask]
    #inputs['T'] = inputs['T'][mask]
    #inputs['cs'] = inputs['cs'][mask]
    

    # -----------------------------------------------------------------------------------------
    # In this part of the function we will remove those sessions which introduce new contrast 
    # levels which were not present in the first sessions (since we want to assess the learning
    # of mice which were always subjected to the same task)
    # -----------------------------------------------------------------------------------------

    # store the part of the expref which indicates a session
    session_array = sessions.to_numpy()
    session_array = np.array([session_array[i][0:12] for i in range(len(session_array))])

    # find the length of each session
    session_lengths = np.array([sum(1 for i in g) for k, g in groupby(session_array)])
    # create a new entry in the input dictionary with the session lengths
    data['dayLength'] = session_lengths

    # these are the indices of where each session starts (and the last entry is the end of the last session)
    session_boundaries = np.cumsum(session_lengths, dtype=int)

    # this stores the contrast levels which were presented in the first session
    first_unique_contrasts = np.unique(data['c'][0:session_boundaries[0]])
    # loops over sessions 'in order', to find the first which has a different 
    # set of contrast levels from those of the first session
    max_sessions = 1
    for i in range(len(session_boundaries) - 1):
        unique_contrasts = np.unique(data['c'][session_boundaries[i]:session_boundaries[i + 1]])
        if not np.array_equal(unique_contrasts, first_unique_contrasts):
            break
        else:
            max_sessions += 1

    # only keep the input data corresponding to sessions where
    # no new contrasts were added to those of the first session
    data['r'] = data['r'][:session_boundaries[max_sessions - 1]]
    data['T'] = data['T'][:session_boundaries[max_sessions - 1]]
    data['c'] = data['c'][:session_boundaries[max_sessions - 1]]
    data['dayLength'] = session_lengths[:max_sessions]
    data['indices'] = np.arange(len(data['c']))

    return data


def fitMice(mouse_names, session_boundaries=True, iteration=0):
    """Fits the LDDM model to each of the mice in mouse_names and saves the weights as .npz files.

    Args:
        mouse_names : list of strings, names of the mice to fit.
        session_boundaries: bool, fit allows higher gaussian walk variances at session boundaries (True)
        or fit ignores session boundaries and fits all trials as a continuous dataset (False).
        iteration : int, number indicating the fit iteration (e.g., if you have tried to fit these mice
        before, but maybe with a different initial point in the optimisation, you can indicate this with
        this parameter).
    Returns:
        None : it saves the save_dict for each mouse as .npz files.
    """
    
    if type(mouse_names) == str:
        data = readTable(mouse_names)
        fitBehavData(data, mouse_name=mouse_names, session_boundaries=session_boundaries, hess_calc='weights', save=True, iteration=iteration) # TODO: might change hess_calc to All
        
    elif type(mouse_names) in [list, np.ndarray]:
        for mouse_name in mouse_names:
            data = readTable(mouse_name)
            fitBehavData(data, mouse_name=mouse_name, session_boundaries=session_boundaries, hess_calc='weights', save=True, iteration=iteration) # TODO: might change hess_calc to All
    else:
         raise Exception('mouse_names must be a valid mouse name or a list/array of valid mouse names')


def loadMouseWeights(mouse_name_or_path, path=False, iteration=0):
    """Loads the weights from .npz files into numpy array and also returns dictionary with their names.

    Args:
        mouse_name_or_path : string, name of mouse whose weights you want to load (only if using default N), or the path to the .npz file (for all other cases).
    Returns:
        w : numpy array, contains the weights per trial for mouse in question.
        weight_dict: dict, contains names (and order) of weights as keys and per-trial dimension as values.
    """

    if path:
        mouse_fit = np.load(mouse_name_or_path, allow_pickle=True)[()]
    else:
        mouse_fit_path = os.path.join('mouse_fits', mouse_name_or_path) + '_fit_' + 'i_' + str(iteration) + '.npy'
        mouse_fit = np.load(mouse_fit_path, allow_pickle=True)[()]

    return mouse_fit

def loadSimulatedWeights(simulated_mouse_name_or_path, path=False, iteration=0):
    """Loads the weights from .npz files into numpy array and also returns dictionary with their names.

    Args:
        mouse_name_or_path : string, name of mouse whose weights you want to load (only if using default N), or the path to the .npz file (for all other cases).
    Returns:
        w : numpy array, contains the weights per trial for mouse in question.
        weight_dict: dict, contains names (and order) of weights as keys and per-trial dimension as values.
    """

    if path:
        artificial_mouse_weights = np.load(simulated_mouse_name_or_path, allow_pickle=True)[()]
    else:
        artificial_mouse_weights_path = os.path.join('simulated_weights', simulated_mouse_name_or_path) + '.npy'
        artificial_mouse_weights = np.load(artificial_mouse_weights_path, allow_pickle=True)[()]

    return artificial_mouse_weights






