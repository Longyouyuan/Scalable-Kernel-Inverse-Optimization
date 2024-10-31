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
from cvx_solver import *
from cd_solver import *
import argparse


parser = argparse.ArgumentParser("Hyperparameters Setting for KTIO")
parser.add_argument("--k", type=float, default=0.001 * 1e-3 * 1, help="coefficient of regularization term")
parser.add_argument("--scaler_d_T", type=float, default=200, help="scaler divided by T")
parser.add_argument("--score", type=int, default=0, help="expected score")
parser.add_argument("--env", type=int, default=0, help="dataset")
parser.add_argument("--it", type=int, default=20, help="max iteration")
parser.add_argument("--batch", type=int, default=5000, help="batch for iteration")
args = parser.parse_args()

env_name = ['hopper-expert-v2', 'hopper-medium-v2', 'walker2d-expert-v2', 'walker2d-medium-v2', 'halfcheetah-expert-v2', 'halfcheetah-medium-v2', 'hopper-medium-expert-v2','walker2d-medium-expert-v2','halfcheetah-medium-expert-v2']
env = gym.make(env_name[args.env])

dataset = env.get_dataset()

test = False

# Choose dataset
if args.env <= 5:
  dataset['observations'] = dataset['observations'][0:50000]
  dataset['actions'] = dataset['actions'][0:50000]
else:
  dataset['observations'] = np.vstack((dataset['observations'][0:50000], dataset['observations'][-50000:]))
  dataset['actions'] = np.vstack((dataset['actions'][0:50000], dataset['actions'][-50000:]))

np.random.seed(11)
env.seed(11)

aug_feature = 1
if aug_feature == 1:
    pf = PolynomialFeatures(2)
    dataset['observations'] = pf.fit_transform(dataset['observations'])
    how = pf.powers_

# normalization - Z-score
s_mean = dataset['observations'].mean(axis=0)
s_std = dataset['observations'].std(axis=0)

min_s = -1000
max_s = 1000
dataset['observations'] = np.clip((dataset['observations']-s_mean)/(s_std+1e-7), min_s, max_s)

s = dataset['observations']
a = dataset['actions']

T = s.shape[0]  # data size
print('datasize', T)
print('env_name:{}, k:{}, scaler/T:{}, datasize:{}, batch:{} *****************************************************'.format(env_name[args.env], args.k, args.scaler_d_T, T, args.batch))

del dataset

s_dim = s[0].shape[0]
a_dim = a[0].shape[0]

M = np.vstack((np.eye(a_dim), -np.eye(a_dim)))
W = np.ones(a_dim*2)
k = args.k
scaler = T * args.scaler_d_T

# # Use SCS to solve the whole problem. If the dataset is too large, you may have memory issue!
# true_Lam_value, true_Gam_value, opt_value = cvx_solver(s, a, M, W, k, scaler)  # use cvxpy to solve
# Gamma = true_Gam_value


#-------------------------------------------My algorithm-----------------------------------#
K = fun_kernel(s)
performance_stuff = [env, pf, s_mean, s_std, min_s, max_s]
if test:
  model = CDUtil2(s,a,M,W,K,k,scaler,performance_stuff,args.score,batch=600,verbose=True)
else:
  model = CDUtil2(s,a,M,W,K,k,scaler,performance_stuff,args.score,max_iter=args.it,batch=int(args.batch),verbose=True)

print('-----------------------------------------------WarmUp--------------------------------------------')

true_Gam_value = np.zeros((a_dim, T))
true_Lam_value = []

for i in range(10):
    true_Lam, true_Gam, _ = cvx_solver(s[i*10000:(i+1)*10000], a[i*10000:(i+1)*10000], M, W, k, scaler, warm_up=True, prop=10)
    true_Gam_value[:,i*10000:(i+1)*10000] = true_Gam
    true_Lam_value += true_Lam

true_Lam_value = np.stack(true_Lam_value, axis=0)

model.Gamma = true_Gam_value
model.Lamda = true_Lam_value
del true_Lam,true_Gam,true_Gam_value,true_Lam_value

print('-----------------------------------------------Fit--------------------------------------------')

Lamda, Gamma = model.fit()

dict = {'opt_val': -1,'value_list': model.value_list, 'reward_list': model.reward_list}
np.save('100K-'+env_name[args.env]+'-'+str(args.batch)+'.npy', dict)

env.close()
