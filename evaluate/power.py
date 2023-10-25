import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore 
from sklearn.metrics import pairwise_distances

from statistics import NormalDist
from tqdm import tqdm

def K_ID(X, Y, gamma=1):
    """
    Forms the kernel matrix K for the two sample test using the SE-T kernel with bandwidth gamma
    where T is the identity operator
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel, if -1 then median heuristic is used to pick gamma
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    n_obs = X.shape[1]
    XY = np.vstack((X,Y))
    dist_mat = (1/np.sqrt(n_obs))*pairwise_distances(XY, metric='euclidean')
    if gamma == -1:
        gamma = np.median(dist_mat[dist_mat > 0])
   
    K = np.exp(-0.5 * (1 / gamma ** 2) * (dist_mat ** 2))
    return K

def MMD_K(K, M, N):
    """
    Calculates the empirical MMD^{2} given a kernel matrix computed from the samples and the sample sizes of each distribution.
    
    Parameters:
    K - kernel matrix of all pairwise kernel values of the two distributions
    M - number of samples from first distribution
    N - number of samples from first distribution
    
    Returns:
    MMDsquared - empirical estimate of MMD^{2}
    """
    
    Kxx = K[:N,:N]
    Kyy = K[N:,N:]
    Kxy = K[:N,N:]
    
    t1 = (1. / (M * (M-1))) * np.sum(Kxx - np.diag(np.diagonal(Kxx)))
    t2 = (2. / (M * N)) * np.sum(Kxy)
    t3 = (1. / (N * (N-1))) * np.sum(Kyy - np.diag(np.diagonal(Kyy)))
    
    MMDsquared = (t1-t2+t3)
    
    return MMDsquared


def two_sample_test(X, Y, hyp, n_perms, z_alpha = 0.05, make_K = K_ID, return_p = False):
    """
    Performs the two sample test and returns an accept or reject statement
    
    Parameters:
    X - (n_samples, n_obs) array of samples from the first distribution 
    Y - (n_samples, n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel
    n_perms - number of permutations performed when bootstrapping the null
    z_alpha - rejection threshold of the test
    return_p - option to return the p-value of the test
    make_K - function called to construct the kernel matrix used to compute the empirical MMD
    
    Returns:
    rej - 1 if null rejected, 0 if null accepted
    p-value - p_value of test
    
    """
    
    # Number of samples of each distribution is identified and kernel matrix formed
    M = X.shape[0]
    N = Y.shape[0]

    K = make_K(X, Y, hyp)

    # Empirical MMD^{2} calculated
    MMD_test = MMD_K(K, M, N)
    
    # For n_perms repeats the kernel matrix is shuffled and empirical MMD^{2} recomputed
    # to simulate the null
    shuffled_tests = np.zeros(n_perms)
    for i in range(n_perms):
            idx = np.random.permutation(M+N)
            K = K[idx, idx[:, None]]
            shuffled_tests[i] = MMD_K(K,M,N)
    
    # Threshold of the null calculated and test is rejected if empirical MMD^{2} of the data
    # is larger than the threshold
    q = np.quantile(shuffled_tests, 1.0-z_alpha)
    rej = int(MMD_test > q)
    
    if return_p:
        p_value = 1-(percentileofscore(shuffled_tests,MMD_test)/100)
        return rej, p_value
    else:
        return rej


def _power_test(X_samples,Y_samples,gamma,n_tests,n_perms,z_alpha = 0.05,make_K = K_ID,return_p = False):
    """
    Computes multiple two-sample tests and returns the rejection rate
    
    Parameters:
    X_samples - (n_samples*n_tests,n_obs) array of samples from the first distribution 
    Y_samples - (n_samples*n_tests,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel
    n_tests - number of tests to perform
    n_perms - number of permutations performed when bootstrapping the null
    z_alpha - rejection threshold of the test
    make_K - function called to construct the kernel matrix used to compute the empirical MMD
    return_p - option to return the p-value of the test
    
    Returns:
    power - the rate of rejection of the null
    """
    
    # Number of samples of each distribution is identified
    M = int(X_samples.shape[0]/n_tests)
    N = int(Y_samples.shape[0]/n_tests)
    rej = np.zeros(n_tests)

    # For each test, extract the data to use and then perform the two-sample test
    for t in range(n_tests):
        X_t = np.array(X_samples[t*M:(t+1)*M,:])
        Y_t = np.array(Y_samples[t*N:(t+1)*N,:])
        rej[t] = two_sample_test(X_t,Y_t,gamma,n_perms,z_alpha = z_alpha,make_K = make_K,return_p = return_p)

    # Compute average rate of rejection
    power = np.mean(rej)
    return power


def plot_reject_null_hyp(X_samples, Y_samples, rejection):
    reject = 'reject' if int(rejection) == 1 else 'not reject'
    print(reject)
    plt.figure(figsize=(12,4))
    X_t_label = 'Ground truth'
    Y_t_label = 'Generation'

    M = X_samples.shape[0]
    domain = np.linspace(-10, 10, 100)
    for i in range(M):
        plt.plot(domain, X_samples[i, :],'-',color='red', label=X_t_label)
        plt.plot(domain, Y_samples[i, :],'-',color='blue', label=Y_t_label)
        X_t_label = '_nolegend_'
        Y_t_label = '_nolegend_'
    plt.legend(loc='upper right')  
    plt.ylim([-105, 105]) 
    plt.show()


def power_test(y_0, y_t, n_tests=100, n_perms=100, gamma=-1):
    X = y_0.cpu().squeeze(-1)  # ground truth -> reduce dimension [B, N, D] -> [B, N]
    Y = y_t.cpu().squeeze(-1)  # prediction -> redice dimension [B, N, D] -> [B, N]

    n_tests = X.shape[0] // 10
    gamma = -1

    power = _power_test(X_samples=X, Y_samples=Y, gamma=gamma, n_tests=n_tests, n_perms=n_perms)
    # print(f'Power = {power * 100:.5f}%')
    return power * 100

def confidence_interval(data, N=10, confidence=0.95):
    '''
    Computes confidence interval (default 95% => confidence=0.95)
    Parameters:
     - data: list input of data
     - N: number of data
     - confidence: confidence percent
    '''
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = dist.stdev * z / ((N - 1) ** 0.5)
    return dist.mean, h

def calculate_ci(y1, y0, gamma=-1, n_tests=10, n_perms=1000, N=30):
    n_tests = y1.shape[0] // 10
    n_perms = 1000
    F = []
    for _ in tqdm(range(N)):
        F.append(power_test(y1, y0, gamma=gamma, n_tests=n_tests, n_perms=n_perms))
    mean, interval = confidence_interval(F, N)
    test_power_interval = str(np.round(mean, 3)) + u"\u00B1" + str(np.round(interval, 3))
    return test_power_interval
