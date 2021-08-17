import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap, hessian
from jax.scipy.stats.norm import logcdf as jaxlogcdf
from jax.scipy.stats.norm import cdf as jaxcdf
from scipy.optimize import minimize
from helper.memoize import memoize
from helper.jacHessCheck import jacHessCheck, jacEltsCheck
from helper.helperFunctions import (
    DT_X_D,
    sparse_logdet,
    make_invSigma,
    myblk_diags,
)

from jax.config import config
config.update("jax_enable_x64", True)  # enable higher FP precision for jax


def getMAP(dat, hyper, weights, method=None, E0=None, showOpt=0):
    """Estimates MAP parameters with a random walk prior.
    Args:
        dat : dict, all data from a specific subject
        hyper : dict, a dictionary of hyperparameters used to construct the prior
            Must at least include sigma, can also include sigInit, sigDay
        weights : dict, name and count of weights to fit
        method : str, control over type of learning, defaults to standard
            trial-by-trial fitting; '_days' and '_constant' also supported
        E0 : np.array, initial parameter estimate, must be of appropriate size N*K,
            defaults to 0.01
        showOpt : int, {0 : no text, 1 : verbose, 2+ : Hess + deriv check, done
            showOpt-1 times}
    Returns:
        hess : dict, dictionary of sparse matrices needed to construct the Hessian of the log posterior at eMode,
            used for Laplace appx. in evidence maximisation.
        logEvd : float, log of the evidence
        llstruct : dict, dictionary containing the components of the log evidence and other info
    """

    # -----
    # Initializations and Sanity Checks
    # -----
    # Check and count trials
    if 'inputs' not in dat or 'r' not in dat or 'T' not in dat or type(
            dat['inputs']) is not dict:
        raise Exception('getMAP_PBups: insufficient input, missing r, or missing T')
    N = len(dat['r'])

    # Check validity of 'r', must be 0 (left choice) and 1 (right choice)
    if not onp.array_equal(onp.unique(dat['r']), [0, 1]):
        raise Exception('getMAP_PBups: r must be parametrized as 0 and 1 only.')

    # Check and count weights
    K = 0
    if type(weights) is not dict:
        raise Exception('weights must be a dict')
    for i in weights.keys():
        if type(weights[i]) is not int or weights[i] < 0:
            raise Exception('weight values must be non-negative ints')
        K += weights[i]

    # Check if using constant weights or by-day weights
    if method is None:
        w_N = N
    elif method == '_constant':
        w_N = 1
    elif method == '_days':
        w_N = len(dat['dayLength'])
    else:
        raise Exception('method type ' + method + ' not supported')

    # Initialize weights to particular values (default 0.01)
    if E0 is not None:
        if type(E0) is not onp.ndarray:
            raise Exception('E0 must be an array, not' + str(type(E0)))

        if E0.shape == (w_N * K,):
            eInit = E0.copy()
        elif E0.shape == (K, w_N):
            eInit = E0.flatten()
        else:
            raise Exception('E0 must be shape (w_N*K,) or (K, w_N), not ' +
                            str(E0.shape))
    else:
        eInit = 0.01*onp.ones(w_N * K)  # not used (better to provide custom init in hyperOpt)

    # Do sanity checks on hyperparameters
    if 'sigma' not in hyper:
        raise Exception('WARNING: sigma not specified in hyper dict')
    if 'alpha' in hyper:
        raise Exception('WARNING: alpha is not supported')
    if method == '_constant':
        if 'sigInit' not in hyper or hyper['sigInit'] is None:
            print('WARNING: sigInit being set to sigma for method', method)
    if method == '_days':
        if 'sigDay' not in hyper or hyper['sigDay'] is None:
            print('WARNING: sigDay being set to sigma for method', method)

    # check for correct use of dayLength
    if ('dayLength' not in dat) and (
            ('sigDay' in hyper and hyper['sigDay'] is not None) or
            (method == '_days')):
        print('WARNING: sigDay has no effect, dayLength not supplied in dat')
        dat['dayLength'] = onp.array([], dtype=int)

    # Account for missing trials from running xval (i.e. gaps from test set)
    if 'missing_trials' in dat and dat['missing_trials'] is not None:
        if len(dat['missing_trials']) != N:
            raise Exception('missing_trials must be length N if used')
    else:
        dat['missing_trials'] = None

    # -----
    # MAP estimate
    # -----

    # Prepare minimization of loss function, Memoize to preserve Jac+Hess info
    lossfun = memoize(negLogPost)
    my_args = (dat, hyper, method)

    if showOpt:
        opts = {'disp': True} #'gtol': 1e-9, 'inexact': False}
        callback = print
    else:
        opts = {'disp': False} #'gtol': 1e-9, 'inexact': False}
        callback = None

    # Actual optimization call
    # Uses 'hessp' to pass a function that calculates 
    # product of Hessian with arbitrary vector
    if showOpt:
        print('Obtaining MAP estimate...')
    result = minimize(
        lossfun,
        eInit,
        jac=lossfun.jacobian,
        hessp=lossfun.hessian_prod,  # todo: look at jax implementation of hessp
        method='trust-ncg',
        tol=1e-9,
        args=my_args,
        options=opts,
        callback=callback,
    )

    # Recover the results of the optimization
    eMode = result.x

    # Print message if optimizer does not converge (usually still pretty good)
    if showOpt and not result.success:
        print('WARNING — MAP estimate: minimize() did not converge\n',
              result.message)
        print('NOTE: this is usually irrelevant as the optimizer still finds '
              'a good solution. If you are concerned, run a check of the '
              'Hessian by setting showOpt >= 2')

    # Run DerivCheck & HessCheck at eMode (will run ShowOpt-1 distinct times)
    if showOpt >= 2:
        print('** Jacobian and Hessian Check **')
        for check in range(showOpt - 1):
            print('\nCheck', check + 1, ':')
            jacHessCheck(lossfun, eMode, *my_args)
            print('')
            
        print('** Jacobian check **')
        for check in range(showOpt - 1):
            print('\nCheck', check + 1, ':')
            jacEltsCheck(lossfun, 2, eMode, *my_args)
            print('')

    # -----
    # Evidence (Marginal likelihood)
    # -----

    # Prior and likelihood at eMode
    if showOpt:
        print('Calculating evd, first prior and likelihood at eMode...')
    pT, lT = getPosteriorTerms(eMode, *my_args)

    # this will be returned to be used in the Laplace Appx.
    hess = {'H': lT['ddlogli']['H'], 'K': lT['ddlogli']['K'], 'ddlogprior': pT['ddlogprior']}

    # Posterior term (with Laplace approx), calculating sparse log determinant
    if showOpt:
        print('Now the posterior with Laplace approx...')
    center = -pT['ddlogprior'] - lT['ddlogli']['H']  # lambdaMAP
    logterm_post = (1 / 2) * sparse_logdet(center)

    # Compute Log evd and construct dict of likelihood, prior,
    # and posterior terms
    logEvd = lT['logli'] + pT['logprior'] - logterm_post
    if showOpt:
        print('Evidence:', logEvd)

    # Package up important terms to return
    llstruct = {'lT': lT, 'pT': pT, 'eMode': eMode}

    return hess, logEvd, llstruct


def negLogPost(*args):
    """Returns negative log posterior (and its first and second derivative)
    Intermediary function to allow for getPosteriorTerms to be optimized
    Args:
        same as getPosteriorTerms()
    Returns:
        negL : negative log-posterior
        dL : 1st derivative of the negative log-posterior
        ddL : 2nd derivative of the negative log-posterior,
            kept as a dict of sparse terms!
    """

    # Get prior and likelihood terms
    [priorTerms, liTerms] = getPosteriorTerms(*args)  # pylint: disable=no-value-for-parameter

    # Negative log posterior
    negPost = -priorTerms['logprior'] - liTerms['logli']
    negdPost = -priorTerms['dlogprior'] - liTerms['dlogli']
    negddPost = {'negddlogprior': -priorTerms['ddlogprior'], 'negH': -liTerms['ddlogli']['H'], 'K': liTerms['ddlogli']['K']}

    return negPost, negdPost, negddPost


def getPosteriorTerms(E_flat, dat, hyper, method=None):
    """Given a sequence of parameters formatted as an N*K vector, calculates
    random-walk log priors & likelihoods and their derivatives
    Args:
        E_flat : array, the N*K parameters, flattened to a single vector
        ** all other args are same as in getMAP **
    Returns:
        priorTerms : dict, the log-prior as well as 1st + 2nd derivatives
        liTerms : dict, the log-likelihood as well as 1st + 2nd derivatives
    """

    # !!! TEMPORARY --- Need to update !!!
    if method in ['_days', '_constant']:
        raise Exception(
            'Need efficient calculations for _constant or _days methods')

    # ---
    # Initialization
    # ---

    # If function is called directly instead of through getMAP,
    # fill in dummy values
    if 'dayLength' not in dat:
        dat['dayLength'] = onp.array([], dtype=int)
    if 'missing_trials' not in dat:
        dat['missing_trials'] = None

    # useful values
    N = len(dat['r'])
    K = int(len(E_flat)/N)

    # Determine type of analysis (standard, constant, or day weights)
    if method is None:
        w_N = N
        # the first trial index of each new day
        days = onp.cumsum(dat['dayLength'], dtype=int)[:-1]
        missing_trials = dat['missing_trials']
    elif method == '_constant':  # trains the weights so that they explain all the trials
        w_N = 1
        days = onp.array([], dtype=int)
        missing_trials = None
    elif method == '_days':  # trains the weights so that only change across days
        w_N = len(dat['dayLength'])
        days = onp.arange(1, w_N, dtype=int)
        missing_trials = None
    else:
        raise Exception('method ' + method + ' not supported')

    # Check shape of E_flat, with
    # w_N (effective # of trials) * K (# of weights) elements
    if E_flat.shape != (w_N * K,):
        print(E_flat.shape, w_N, K, method)
        raise Exception('parameter dimension mismatch (#trials * #weights)')

    # ---
    # Construct random-walk prior, calculate priorTerms
    # ---
  
    # Construct random walk covariance matrix Sigma^-1, use sparsity for speed
    invSigma = make_invSigma(hyper, days, missing_trials, w_N, K)
    invC = DT_X_D(invSigma, K)  # transform by difference matrix to get invC 

    # Calculate the log-determinant of prior covariance,
    # the log-prior, 1st, & 2nd derivatives
       
    logdet_C = -onp.sum(np.log(invSigma.diagonal()))
    
    logprior = (1 / 2) * (-logdet_C - E_flat @ invC @ E_flat)
    dlogprior = -invC @ E_flat
    ddlogprior = -invC

    priorTerms = {
        'logprior': logprior,
        'dlogprior': dlogprior,
        'ddlogprior': ddlogprior,
    }

    # ---
    # Construct likelihood, calculate liTerms
    # ---

    # we fix the noise values to a reasonable value given the scale of the data
    sig_o = 1
    sig_i = 0.01

    # reshaping E_flat for vmap in ll_hessian_blks
    E = onp.reshape(E_flat, (K, w_N), order='C')

    # hessian of likelihood calculated using jax
    HlliList = ll_hessian_blks(E, dat, sig_o, sig_i)
  

    # INSERT CODE HERE TO HANDLE _days OR _constant METHODS


    # Calculate the log-likelihood and 1st&2nd derivatives
  
    logli = log_likelihood(E, dat, sig_o, sig_i)
    dlogli = grad(log_likelihood)(E, dat, sig_o, sig_i).flatten()
    ddlogli = {'H': myblk_diags(HlliList), 'K': K}

    # store the results in a dict
    liTerms = {'logli': logli, 'dlogli': dlogli, 'ddlogli': ddlogli}

    return priorTerms, liTerms

@jit
def log_inv_gauss_pdf(thr, drift, v, t):
    
    A = np.log(thr / np.sqrt(2 * np.pi * v * t ** 3))
    
    return A - (thr - drift * t) ** 2 / (2 * v * t)

@jit
def inv_gauss_cdf(thr, drift, v, t):

    # A and logB are separated for numerical stability
    A = jaxcdf((drift * t - thr) / np.sqrt(v * t))
    logB = 2. * thr * (drift / v) + jaxlogcdf(-(drift * t + thr) / np.sqrt(v * t))
    
    return A + np.exp(logB)

@jit
def log_likelihood(E, dat, sig_o, sig_i):

    # this if/elif is required due to the vmap in the jax function
    # which inputs the weights as vectors for each trial 
    # as opposed to the parameter x trial matrix
    
    if E.ndim == 2:
        w1 = E[0, :]
        w2 = E[1, :]
        b1 = E[2, :]
        b2 = E[3, :]
        z = E[4, :]

    elif E.ndim == 1:
        w1 = E[0]
        w2 = E[1]
        b1 = E[2]
        b2 = E[3]
        z = E[4]

    # Calculate drift rates & noise variances for each integrator
    drift1 = w1 * (dat['inputs']['c'] * (dat['inputs']['c'] > 0)) + b1
    drift2 = w2 * (-dat['inputs']['c'] * (-dat['inputs']['c'] > 0)) + b2

    # variances
    v1 = w1 ** 2 * sig_i ** 2 + sig_o ** 2
    v2 = w2 ** 2 * sig_i ** 2 + sig_o ** 2

    # Drift & variance of chosen option
    drift_k = dat['r'] * drift1 + (1 - dat['r']) * drift2
    v_k = dat['r'] * v1 + (1 - dat['r']) * v2

    # Drift & var of unchosen option
    drift_kbar = (1 - dat['r']) * drift1 + dat['r'] * drift2
    v_kbar = (1 - dat['r']) * v1 + dat['r'] * v2

    # Calculate resulting log likelihood
    ll = log_inv_gauss_pdf(z, drift_k, v_k, dat['T'])
    ll2 = np.log(1. - inv_gauss_cdf(z, drift_kbar, v_kbar, dat['T']))

    return np.sum(ll + ll2)

@jit
def ll_hessian_blks(E, dat, sig_o, sig_i):  # todo: can I remove everything but E?
    h = hessian(log_likelihood, 0)  # see jax docs for how this works, in brief, this is the hessian function
    # which we evaluate at w_map where this is done on a trial by trial basis
    hblks = vmap(h, (1, {'T': 0, 'dayLength': None, 'inputs': {'c': 0}, 'missing_trials': None, 'r': 0}, None, None))
    return hblks(E, dat, sig_o, sig_i)
