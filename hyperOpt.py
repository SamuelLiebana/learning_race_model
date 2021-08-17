import numpy as np
from scipy.optimize import minimize
from scipy.sparse import linalg
from getMAP import getMAP, getPosteriorTerms
from helper.invBlkTriDiag import getCredibleInterval
from helper.jacHessCheck import compHess
from helper.helperFunctions import (
    DT_X_D,
    make_invSigma,
    sparse_logdet,
)


def hyperOpt(dat, hyper, weights, optList, method=None, showOpt=0, jump=2,
             hess_calc="weights"):
    """Optimizes for hyperparameters and weights.

    Given data and set of hyperparameters, uses decoupled Laplace to find the
    optimal hyperparameter values (i.e. the sigmas that maximize evidence).
    Args:
        dat : dict, all data from a specific subject
        hyper : dict, hyperparameters and initial values used to construct prior
            Must at least include sigma, can also include sigInit, sigDay
        weights : dict, name and count of which weights to fit.
        optList : list, hyperparameters in 'hyper' to be optimized
        method : str, control over type of fit,
            None is standard, '_days' (fit model per-session) and
            '_constant' (fit model for entire dataset) also supported
        showOpt : int, 0 : no text, 1 : verbose,
            2+ : Hess + deriv check, done showOpt-1 times
        jump : int, how many times the alg can find suboptimal evd before quit
        hess_calc : str, what hessian calculations to perform. If None, then
            the hessian of the weights is returned (standard error on weights
            can be calculated post-hoc). If 'weights', the the standard error
            of the weights is calculated internally and returned instead of the
            hessian. If 'hyper', then standard errors of the hyperparameters is
            computed with the numerical hessian and returned. If 'All', then
            a dict with standard errors for both the weights and the
            hyperparameters is returned.
    Returns:
        best_hyper : hyperparameter values that maximizes evidence of data
        best_logEvd : log-evidence associated with optimal hyperparameter values
        best_eMode : the MAP weights found using best_hyper, maximizing logEvd
        hess_info : dict, Hessian info, specific info depending on `hess_calc`
    """

    # Initialization of optimization
    opt_keywords = {
        'dat': dat,
        'hyper': hyper,
        'weights': weights,
        'optList': optList,
        'method': method,
    }

    current_hyper = hyper.copy()
    best_logEvd = None

    # Make sure all hyperparameters to be optimized are actually provided
    for val in optList:
        if (val not in hyper) or (hyper[val] is None):
            raise Exception('cannot optimize a hyperparameter which has not been given')

    # -----
    # Hyperparameter Optimization
    # -----

    current_jump = jump
    while True:

        if best_logEvd is None:
            # initialising the parameters
            E0 = np.array([np.linspace(0.05, 2, len(dat['r'])), np.linspace(0.05, 2, len(dat['r'])),
                          np.linspace(0.4, 0.7, len(dat['r'])), np.linspace(0.3, 0.5, len(dat['r'])),
                          np.linspace(1, 1, len(dat['r']))])

        else:
            # if not first loop, use the estimate of w_MAP from the last loop
            E0 = llstruct['eMode']  # pylint: disable=used-before-assignment

        # First get MAP for initial hyperparameter setting
        Hess, logEvd, llstruct = getMAP(
            dat,
            current_hyper,
            weights,
            E0=E0,
            method=method,
            showOpt=0)

        # Update best variables
        if best_logEvd is None:
            best_logEvd = logEvd
        if logEvd >= best_logEvd:
            current_jump = jump
            best_hyper = current_hyper.copy()
            best_logEvd = logEvd
            best_Hess = Hess
            best_eMode = llstruct['eMode']
            best_llstruct = llstruct.copy()
        else:
            # If a worse logEvd found, reduce jump by one and
            # move hypers to midpoints, keep old bests
            current_jump -= 1
            for val in optList:
                current_hyper.update({
                    val: (current_hyper[val] + best_hyper[val]) / 2
                })

        if showOpt:
            print('\nInitial log-evidence:', np.round(logEvd, 5))
            for val in optList:
                print(val, np.round(np.log2(current_hyper[val]), 4))

        # Halt optimization early if evidence was worse 'jump'-times in a row
        if not current_jump:
            # Reset optVals and opt_keywords to state when best logEvd
            optVals = []
            for val in optList:
                if np.isscalar(best_hyper[val]):
                    optVals += [np.log2(best_hyper[val])]                   
                else:
                    optVals += np.log2(best_hyper[val]).tolist()                   

            K = best_llstruct['lT']['ddlogli']['K']
            H = best_llstruct['lT']['ddlogli']['H']
            ddlogprior = best_llstruct['pT']['ddlogprior']
            eMode = best_llstruct['eMode']

            # this is part of the w_MAP update with changing hypers
            # which remains fixed as it comes from the likelihood
            LL_v = -(H + ddlogprior)@eMode
            
            opt_keywords.update({
                'LL_terms': best_llstruct['lT']['ddlogli'],  # this contains the hessian of LL
                'LL_v': LL_v,
                'eMode': eMode
            })

            if showOpt:
                print('Optimization halted early due to no improvement in evidence.')
            break

        # Now decouple prior terms from likelihood terms and store values
        K = llstruct['lT']['ddlogli']['K']
        H = llstruct['lT']['ddlogli']['H']
        ddlogprior = llstruct['pT']['ddlogprior']
        eMode = llstruct['eMode']

        # this is part of the w_MAP update with changing hypers
        # which remains fixed as it comes from the likelihood
        LL_v = -(H + ddlogprior)@eMode

        opt_keywords.update({
            'LL_terms': llstruct['lT']['ddlogli'],
            'LL_v': LL_v,
            'eMode': eMode
        })

        # Optimize over hyperparameters
        if showOpt:
            print('\nStarting hyperparameter optimization...')
            opts = {'disp': True, 'maxiter': 15}
            callback = print
        else:
            opts = {'disp': False, 'maxiter': 15}
            callback = None

        # Do hyperparameter optimization in log2
        optVals = []
        
        for val in optList:
            if np.isscalar(current_hyper[val]):
                optVals += [np.log2(current_hyper[val])]
              
            else:
                optVals += np.log2(current_hyper[val]).tolist()

        # perform the minimisation
        result = minimize(
            hyperOpt_lossfun,
            optVals,
            args=opt_keywords,
            method='BFGS',
            options=opts,
            callback=callback,
        )

        # distance between the initial hyperparams and those
        # after the optimisation as a fraction of original values
        diff = np.linalg.norm((optVals - np.array(result.x)) / optVals)
        
        if showOpt:
            print('\nRecovered hyperparameters:', np.array(result.x))
            print('\nRecovered log-evidence:', np.round(-result.fun, 5))
            print('\nDifference in hyperparameters:', np.round(diff, 4))

        # only continue with optimisation if difference is significant
        if diff > 0.1:
            count = 0
            for val in optList:
                if np.isscalar(current_hyper[val]):
                    current_hyper.update({val: 2 ** np.array(result.x[count])})
                    count += 1
                else:
                    current_hyper.update({val: 2 ** np.array(result.x[count:count + K])})
                    count += K
                if showOpt:
                    print(val, np.round(np.log2(current_hyper[val]), 4)) 
                   
        else:
            break

    # After hyperparameters converged, calculate standard error of weights
    # and/or hyperparameters if necessary
    hess_info = {'hess': best_Hess}
    if hess_calc in ['weights', 'All']:
        W_std = getCredibleInterval(best_Hess)
        hess_info.update({'W_std': W_std})
    if hess_calc in ['hyper', 'All']:
        numerical_hess = compHess(fun=hyperOpt_lossfun,
                                  x0=np.array(optVals),
                                  dx=0.01,  # todo: might have to change this
                                  kwargs={'keywords': opt_keywords})
        hyp_std = np.sqrt(np.diag(np.linalg.inv(numerical_hess[0])))
        hess_info.update({'hyp_std': hyp_std})

    return best_hyper, best_logEvd, best_eMode, hess_info


def hyperOpt_lossfun(optVals, keywords):
    """Loss function used by decoupled Laplace to optimize for evidence over
    changes in hyperparameters.
    Args:
        optVals : hyperparameter values currently used to calculate approximate
            evidence, corresponds to hyperparameters listed in OptList
        keywords : dictionary of other values needed for optimization
    Returns:
        evd : the negative log-evidence (to be minimized)
    """

    # Recover N & K
    N = keywords['dat']['r'].shape[0]
    K = keywords['LL_terms']['K']
    method = keywords['method']
    dat = keywords['dat']
    weights = keywords['weights']

    # Reconstruct the prior covariance
    hyper = keywords['hyper'].copy()
    
    count = 0
    for val in keywords['optList']:
        if np.isscalar(hyper[val]):
            hyper.update({val: 2 ** optVals[count]})
            count += 1
        else:
            hyper.update({val: 2 ** optVals[count:count + K]})
            count += K
            
    # Determine type of analysis (standard, constant, or day weights)
    if method is None:
        w_N = N
        days = np.cumsum(dat['dayLength'], dtype=int)[:-1]  # the first trial index of each new day/session
        missing_trials = dat['missing_trials']  # this is None for our dset
    elif method == '_constant':
        w_N = 1
        days = np.array([], dtype=int)
        missing_trials = None
    elif method == '_days':
        w_N = len(dat['dayLength'])
        days = np.arange(1, w_N, dtype=int)
        missing_trials = None
    else:
        raise Exception('method ' + method + ' not supported')
    
    # start by calcuating covariance matrix of
    # prior, and its hessian
    invSigma = make_invSigma(hyper, days, missing_trials, w_N, K)
    ddlogprior = -DT_X_D(invSigma, K)

    # Retrieve terms for decoupled Laplace appx.
    H = keywords['LL_terms']['H']
    LL_v = keywords['LL_v']
    Lambda_MAP_inv = -H - ddlogprior

    # Decoupled Laplace appx of new weights given new sigma
    E_flat = linalg.spsolve(Lambda_MAP_inv, LL_v)
    
    #print('isstop', istop)

    # Calculate likelihood and prior terms with new E_flat = w_MAP(\theta_{ME})
    pT, lT = getPosteriorTerms(E_flat, dat=dat, hyper=hyper, method=method)

    # Calculate posterior term, then approximate evidence for new sigma
    #logterm_post = (1 / 2) * sparse_logdet(Lambda_MAP_inv) # alternative
    logterm_post = (1 / 2) * sparse_logdet(-ddlogprior - lT['ddlogli']['H'])

    evd = lT['logli'] + pT['logprior'] - logterm_post

    return -evd#/(w_N)#**0.5) # may need to normalise this if numerical instabilities arise