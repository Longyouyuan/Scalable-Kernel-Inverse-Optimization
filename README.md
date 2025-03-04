# Scalable Kernel Inverse Optimization

Official implementation of [Scalable Kernel Inverse Optimization](https://neurips.cc/virtual/2024/poster/95494) paper.

## Installation

```pip install -r requirements.txt```

## Content

-  ```algorithm.py```
This is the main executable python file. 
The dataset used for training and the relevant hyperparameters are specified in this file. 
Subsequently, the SCS/SSO solver is invoked to solve the inverse optimization problem.
* For example, you can run the code with `python algorithm.py --k 1e-6 --scaler_d_T 100 --score 0 --env 6 --it 20 --batch 10000`

- ```cvx_solver.py```
Construct the inverse optimization problem using CVXPY and solve it with the SCS solver.

- ``` cd_solver.py```
Use SSO algorithm to solve the inverse optimization problem in a distributed fashion.
At every iteration, use CVXPY to model the sub problem and use SCS to solve it.

## Citation

```
@article{long2025scalable,
  title={Scalable kernel inverse optimization},
  author={Long, Youyuan and Ok, Tolga and Zattoni Scroccaro, Pedro and Mohajerin Esfahani, Peyman M},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={99464--99487},
  year={2025}
}
```
