import os

# import gymnasium as gym
import d4rl # Import required to register environments, you may need to also import the submodule
import cvxpy as cp
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import gym
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel, chi2_kernel
from quadprog import solve_qp
import time

def fun_kernel(s, Y=None, l=0):
    if l == 0:
        # return polynomial_kernel(s, degree=7, gamma=2)
        return rbf_kernel(s, gamma=None)
        # return sigmoid_kernel(s, gamma=0.1, coef0=0)
        # return chi2_kernel(s, gamma=1.0)
        # return s @ s.T
    else:
        # return polynomial_kernel(s, Y=Y, degree=7, gamma=2)
        return rbf_kernel(s, Y=Y, gamma=None)
        # return sigmoid_kernel(s, Y=Y, gamma=0.1, coef0=0)
        # return chi2_kernel(s, Y=Y, gamma=1.0)
        # return s @ Y.T


def cvx_solver(s, a, M, W, k, scaler, Lam_init=None, Gam_init=None, warm_up=False, prop=2):
    T = s.shape[0]
    
    if warm_up == True:
        N = prop*T
    else:
        N = T
    
    s_dim = s[0].shape[0]
    a_dim = a[0].shape[0]

    start_time = time.time()
    Lam = [cp.Variable((a_dim, a_dim), symmetric=True) for _ in range(T)]
    Gam = cp.Variable((a_dim, T))

    if Lam_init != None:
        for i in range(len(Lam)):
            Lam[i].value = Lam_init[i]
        Gam.value = Gam_init

    K = fun_kernel(s)
    K = K / (scaler * k)
    K_kron = np.kron(K, np.eye(a_dim))
    K_kron += np.eye(K_kron.shape[0]) * 1e-7

    obj = cp.quad_form(cp.vec(a * scaler / N - 2 * Gam.T, order='C'), cp.psd_wrap(K_kron)) + cp.trace(cp.sum(Lam))
    del K_kron

    alpha = np.eye(1) * scaler / (4 * N)
    cons = [cp.bmat([[Lam[t], Gam[:, t:t + 1]],
                     [Gam[:, t:t + 1].T, alpha]]) >> 0 for t in range(T)]
    cons += [np.array([W / N]).T * scaler - 2 * M @ Gam >= 0]

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(verbose=True, solver="SCS")

    print("dual optimal value", prob.value/T)
    Lam_value = [Lam[i].value for i in range(T)]
    Gam_value = Gam.value
    # print("Lam", Lam_value)
    # print("Gam", Gam_value)
    end_time = time.time()
    print("Run time：", (end_time - start_time) / 60, "minutes")

    return Lam_value, Gam_value, prob.value/T
