import os

import cvxpy as cp
import numpy as np
import time
import d4rl # Import required to register environments, you may need to also import the submodule
from sklearn.preprocessing import PolynomialFeatures
import gym
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel, chi2_kernel
from quadprog import solve_qp
from cvx_solver import *


class CDUtil2:
    def __init__(self,s,a,M,W,K,k,scaler,performance_stuff,score,max_iter=20,update_tol=1e-3, batch=1,verbose=False):
        self.s = s
        self.a = a*scaler
        self.M = M
        self.W = W*scaler
        self.K = K/(k*scaler)
        self.k = k*scaler
        self.scaler = scaler
        self.max_iter = max_iter
        self.update_tol = update_tol
        self.s_dim = s[0].shape[0]
        self.a_dim = a[0].shape[0]
        self.T = s.shape[0]
        self.batch = batch
        self.verbose = verbose
        self.score = score

        self.iter = np.ceil(self.T/self.batch).astype(int)
        self.value = np.inf
        self.order = np.arange(self.T)
        self.value_list = []
        self.reward_list = []

        # [env, pf, s_mean, s_std, min_s, max_s]
        self.env = performance_stuff[0]
        self.pf = performance_stuff[1]
        self.s_mean = performance_stuff[2]
        self.s_std = performance_stuff[3]
        self.min_s = performance_stuff[4]
        self.max_s = performance_stuff[5]

        self.obj = np.inf
        self.max_KKT_violate_value = -1
        self.Lamda = np.zeros((self.T, self.a_dim, self.a_dim))  # cache for optimal solution Lamda
        self.Gamma = np.zeros((self.a_dim, self.T))*50  # cache for optimal solution Gamma
        self.cache = (self.a.T / self.T - 2 * self.Gamma)@K/self.k  # precalculate some big matrix

        self.subprob = SubProb(self.a_dim,M,W,scaler,self.T, batch)


    def fit(self):
        # Use SSO to solve the optimization problem

        it = 0
        it_all = 0
        it_heuristic = 0
        entire = False  # go through all the data
        pair_changed = 0
        print('it: {}, it_all:{}, it_heur:{}, obj: {}'.format(it, it_all, it_heuristic, self.calculate_obj()))
        self.value_list.append(self.calculate_obj())
        while it < self.max_iter:
            pair_changed = 0
            if entire:
                for i in range(self.iter):
                    self.take_last_batch()
                    pair_changed += self.take_step()
                    self.reward_list.append(self.performance_test(times=100))
                    it_all += 1
                    it += 1
                    self.value = self.calculate_obj()
                    self.value_list.append(self.value)
                    print('it: {}, it_all:{}, it_heur:{}, obj: {}, reward:{}'.format(it, it_all, it_heuristic, self.value, self.reward_list[-1]))
                self.max_KKT_violate_value = -1
                if pair_changed == 0:
                    print("all do not move much!")
                    break
            else:
                it_heuristic += 1
                it += 1
                self.take_heuristic_batch()  # find some t that violate KKT a lot
                pair_changed += self.take_step()
                self.reward_list.append(self.performance_test(times=100))
                self.value = self.calculate_obj()
                self.value_list.append(self.value)
                print('it: {}, it_all:{}, it_heur:{}, obj: {}, reward:{}'.format(it, it_all, it_heuristic, self.value, self.reward_list[-1]))

            if entire:
                entire = False
            elif pair_changed == 0:
                entire = True

        self.recover()
        return self.Lamda, self.Gamma

    def recover(self):
        # Recover the order of optimizers

        sorted_indices = np.argsort(self.order)
        self.Gamma = self.Gamma[:, sorted_indices]
        self.Lamda = self.Lamda[sorted_indices]

    def take_heuristic_batch(self):
        # Place the selected coordinates in the front row.

        change_indices = self.find_KKT_violater()

        leng = len(change_indices)

        if len(change_indices) < self.batch:
            #  we choose fixed coordinated instead in this case
            available_fixed_numbers = set(range(self.T)) - set(change_indices)
            fixed_numbers = np.random.choice(list(available_fixed_numbers), size=self.T-self.batch, replace=False)
            opt_numbers = list(set(range(self.T)) - set(fixed_numbers))
        else:
            fixed_numbers = list(set(range(self.T)) - set(change_indices))
            opt_numbers = change_indices

        self.s = np.vstack((self.s[opt_numbers], self.s[fixed_numbers]))
        self.a = np.vstack((self.a[opt_numbers], self.a[fixed_numbers]))
        self.Lamda = np.concatenate((self.Lamda[opt_numbers], self.Lamda[fixed_numbers]))
        self.Gamma = np.hstack((self.Gamma[:, opt_numbers], self.Gamma[:, fixed_numbers]))
        self.K = np.vstack((self.K[opt_numbers], self.K[fixed_numbers]))
        self.K = np.hstack((self.K[:, opt_numbers], self.K[:, fixed_numbers]))
        self.order = np.hstack((self.order[opt_numbers], self.order[fixed_numbers]))

    def take_last_batch(self):
        # Place the selected coordinates in the front row.

        self.s = np.vstack((self.s[-self.batch:], self.s[:-self.batch]))
        self.a = np.vstack((self.a[-self.batch:], self.a[:-self.batch]))
        self.Lamda = np.concatenate((self.Lamda[-self.batch:], self.Lamda[:-self.batch]))  # (self.T, self.a_dim, self.a_dim)
        self.Gamma = np.hstack((self.Gamma[:, -self.batch:], self.Gamma[:, :-self.batch]))  # (self.a_dim, self.T)
        self.K = np.vstack((self.K[-self.batch:], self.K[:-self.batch]))
        self.K = np.hstack((self.K[:, -self.batch:], self.K[:, :-self.batch]))
        self.order = np.hstack((self.order[-self.batch:], self.order[:-self.batch]))

    def find_KKT_violater(self):
        # Find the coordinates that violate the KKR condition the most

        tilde_lamda = np.array([self.W / self.T]).T - 2 * self.M @ self.Gamma
        indices = np.where(np.all(tilde_lamda > 0.1, axis=0) == True)[0]  # indices that lamda is zero

        vec = -2*(self.a.T / self.T - 2 * self.Gamma)@self.K[:, indices]
        err = np.abs(np.trace(self.Lamda[indices], axis1=1, axis2=2) + 2 * np.sum(self.Gamma[:, indices] * vec, axis=0) \
                     + self.scaler / (4 * self.T) * np.sum(np.square(vec), axis=0))

        max_err = np.max(err)
        if self.max_KKT_violate_value < 0:  # mean this is a new heuristic term
            self.max_KKT_violate_value = max_err
        elif max_err < self.max_KKT_violate_value / 1000:  # if the max_err is not big, we think we optimize enough here
            return []
        change_indices = indices[np.where(err > np.maximum(max_err*0.1, 0.1))[0]]  # find a batch
        if len(change_indices) > self.batch:
            change_indices = np.random.choice(change_indices, size=self.batch, replace=False)
        # change_indices = [indices[np.argmax(err)]] # find the largest

        return change_indices

    def calculate_obj(self):
        # Calculate current objective value

        m = self.a / self.T - 2 * self.Gamma.T
        obj1 = np.sum((m@m.T)*self.K)/self.T
        obj2 = np.sum(np.trace(self.Lamda, axis1=1, axis2=2))/self.T

        return obj1+obj2

    def take_step(self):
        # Optimize the sub problem with selected coordinates

        K_kron = np.kron(self.K[:self.batch, :self.batch], np.eye(self.a_dim))
        K_kron += np.eye(K_kron.shape[0]) * 1e-5
        mat = self.K[:self.batch, self.batch:] @ (self.a[self.batch:] / self.T - 2 * self.Gamma.T[self.batch:])\
             + self.K[:self.batch, :self.batch] @ (self.a[:self.batch] / self.T)
        Lamda_t, Gamma_t, self.value = self.subprob.solve(K_kron, mat, verbose=self.verbose)

        change = np.sum(np.square(self.Gamma[:, :self.batch] - Gamma_t))
        if change < 1e-5:
            return 0

        self.Lamda[:self.batch] = Lamda_t
        self.Gamma[:, :self.batch] = Gamma_t

        return 1

    def solve_subprob(self):
        # Not used in this case

        Lam_var = [cp.Variable((self.a_dim, self.a_dim), symmetric=True) for _ in range(self.batch)]
        Gam_var = cp.Variable((self.a_dim, self.batch))

        K_kron = np.kron(self.K, np.eye(self.a_dim))
        K_kron += np.eye(K_kron.shape[0]) * 1e-7

        Gam = cp.hstack((Gam_var, self.Gamma[:, self.batch:]))
        obj = cp.quad_form(cp.vec(self.a / self.T - 2 * Gam.T, order='C'), cp.psd_wrap(K_kron)) \
              + cp.trace(cp.sum(Lam_var))

        alpha = np.eye(1) * self.scaler / (4 * self.T)
        cons = [cp.bmat([[Lam_var[t], Gam_var[:, t:t + 1]],
                         [Gam_var[:, t:t + 1].T, alpha]]) >> 0 for t in range(self.batch)]
        cons += [np.array([self.W / self.T]).T - 2 * self.M @ Gam_var >= 0]

        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(verbose=self.verbose, solver="SCS")

        Lam_value = [Lam_var[i].value for i in range(self.batch)]
        return Lam_value, Gam_var.value

    def performance_test(self, times=100):
        # Test performance under current solution

        total_reward = 0
        total_reward_norm = 0
        epi_reward = []
        epi_steps = []
        for _ in range(times):
            ss = self.env.reset()

            ss = np.clip((self.pf.fit_transform(np.array([ss])) - self.s_mean) / (self.s_std + 1e-7), self.min_s, self.max_s)

            episode_reward = 0
            steps = 0
            while True:
                # env.render()
                # time.sleep(0.001)
                K = fun_kernel(np.array(ss), Y=self.s, l=1)
                f = -(K / self.k) @ (self.a / self.T - 2 * self.Gamma.T)
                action = solve_qp(np.eye(self.a_dim), -f[0].T, -self.M.T, -self.W/self.scaler)[0]
                # a = env.action_space.sample()  # ���ѡ��һ������
                ss_, r, done, info = self.env.step(action)
                steps += 1
                episode_reward += r

                ss = np.clip((self.pf.fit_transform(np.array([ss_])) - self.s_mean) / (self.s_std + 1e-7), self.min_s, self.max_s)

                if done:
                    # print(episode_reward, self.env.get_normalized_score(episode_reward))
                    epi_reward.append(episode_reward)
                    epi_steps.append(steps)
                    total_reward += episode_reward
                    total_reward_norm += self.env.get_normalized_score(episode_reward)
                    break
        return total_reward_norm/times


class SubProb:
    def __init__(self, a_dim,M,W,scaler,T, batch):  # ע���a��ǰ��scaler���� KҲ�Ǳ�����scaler,k��
        self.batch = batch

        self.Kron = cp.Parameter((batch*a_dim, batch*a_dim), symmetric=True)
        self.mat = cp.Parameter((batch, a_dim))

        self.Lam_var = [cp.Variable((a_dim, a_dim), symmetric=True) for _ in range(batch)]
        self.Gam_var = cp.Variable((a_dim, batch))

        obj = 4 * cp.quad_form(cp.vec(self.Gam_var.T, order='C'), cp.psd_wrap(self.Kron)) \
               - 4 * cp.sum(cp.multiply(self.Gam_var.T, self.mat)) \
               + cp.trace(cp.sum(self.Lam_var))

        alpha = np.eye(1) * scaler / (4 * T)
        cons = [cp.bmat([[self.Lam_var[t], self.Gam_var[:, t:t + 1]],
                         [self.Gam_var[:, t:t + 1].T, alpha]]) >> 0 for t in range(self.batch)]
        cons += [np.array([W * scaler / T]).T - 2 * M @ self.Gam_var >= 0]

        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def solve(self, K, mat, verbose=False):
        self.Kron.value = (K+K.T)/2
        self.mat.value = mat

        self.prob.solve(verbose=verbose, solver="SCS", warm_start=False, max_iters=2000)

        Lam_value = [self.Lam_var[i].value for i in range(self.batch)]

        return Lam_value, self.Gam_var.value, self.prob.value/self.batch
        
        
        
class SubProb2:
    # Not used in this case

    def __init__(self, a_dim,M,W,scaler,T, batch):  # ע���a��ǰ��scaler���� KҲ�Ǳ�����scaler,k��
        self.batch = batch

        self.Kron = cp.Parameter((T*a_dim, T*a_dim), symmetric=True)
        self.a = cp.Parameter((T, a_dim))
        self.Lam_var = [cp.Variable((a_dim, a_dim), symmetric=True) for _ in range(batch)]
        self.Gam_var = cp.Variable((a_dim, batch))
        self.Gam_para = cp.Parameter((a_dim, T - batch))

        tt = T

        Gam = cp.hstack((self.Gam_var, self.Gam_para))
        obj = cp.quad_form(cp.vec(self.a / tt - 2 * Gam.T, order='C'),cp.psd_wrap(self.Kron))\
              + cp.trace(cp.sum(self.Lam_var))

        alpha = np.eye(1) * scaler / (4 * tt)
        cons = [cp.bmat([[self.Lam_var[t], self.Gam_var[:, t:t + 1]],
                         [self.Gam_var[:, t:t + 1].T, alpha]]) >> 0 for t in range(batch)]
        cons += [np.array([W / tt]).T * scaler - 2 * M @ self.Gam_var >= 0]

        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def solve(self,K, a, Gam_para, verbose=False):
        self.Kron.value = K
        self.a.value = a
        self.Gam_para.value = Gam_para

        self.prob.solve(verbose=verbose, solver="SCS", warm_start=False, max_iters=2000)

        Lam_value = [self.Lam_var[i].value for i in range(self.batch)]

        return Lam_value, self.Gam_var.value, self.prob.value/self.batch
        


def check_KKT(k,K,W,M,scaler,T,a,true_Gam_value,true_Lam_value,index=None):
    # Check the degree of violation of the KKT conditions for each coordinate.

    a_dim = a[0].shape[0]

    tilde_lamda = np.array([W / T]).T * scaler - 2 * M @ true_Gam_value
    lamda_is_zero_index = np.where(np.all(tilde_lamda > 0.1, axis=0) == True)[0]
    alpha = np.eye(1) * scaler / (4 * T)
    if index==None:
        err_all = []
        for t in lamda_is_zero_index:
            matrix1 = np.bmat([[true_Lam_value[t], true_Gam_value[:, t:t + 1]],
                               [true_Gam_value[:, t:t + 1].T, alpha]])
            vec = -2 * ((a.T * scaler / T - 2 * true_Gam_value) @ K[:, t:t + 1]) / (k * scaler)
            matrix2 = np.bmat([[np.eye(a_dim), vec],
                               [vec.T, np.eye(1) * vec.T @ vec]])
            err = np.trace(matrix1 @ matrix2)
            err_all.append(err)
            # print(err)
        return np.abs(np.array(err_all))
    else:
        matrix1 = np.bmat([[true_Lam_value[index], true_Gam_value[:, index:index + 1]],
                           [true_Gam_value[:, index:index + 1].T, alpha]])
        vec = -2 * ((a.T * scaler / T - 2 * true_Gam_value) @ K[:, index:index + 1]) / (k * scaler)
        matrix2 = np.bmat([[np.eye(a_dim), vec],
                           [vec.T, np.eye(1) * vec.T @ vec]])
        err = np.trace(matrix1 @ matrix2)
        # print(err)
        return err

