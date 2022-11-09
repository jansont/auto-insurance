import pandas as pd

from train import *
from utils import *
from pathlib import Path
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Categorical, Integer, Real



def configure_search_space():

    space = {
                        "learning_rate": {
                                            "type": "real",
                                            "lower_bound": 1e-5 ,
                                            "upper_bound": 1e-2, 
                                            "prior": "log-uniform", 
                                            "init": 1e-3
                                        },
                        "iterations" : {
                                            "type": "integer",
                                            "lower_bound": 1e2,
                                            "upper_bound": 5e3,
                                            "prior": "log-uniform",
                                            "init": 1e2
                                        },
                        "l2_leaf_reg": {
                                            "type": "real", 
                                            "lower_bound": 1e-3, 
                                            "upper_bound": 1e2, 
                                            "prior": "log-uniform", 
                                            "init": 1e-1
                                        },
                        "bagging_temperature": {
                                            "type": "real", 
                                            "lower_bound": 1e-3, 
                                            "upper_bound": 1e2, 
                                            "prior": "log-uniform", 
                                            "init": 1
                                        }, 
                        "depth": {
                                            "type": "integer", 
                                            "lower_bound": 1, 
                                            "upper_bound": 12, 
                                            "prior": "uniform", 
                                            "init": 10
                                } 
                    }
    search_space = list()
    x_init = list()

    for key, params in space.items():
        if params["type"] == "real" or params["type"] == "integer":
            class_ = Real if params["type"] == "real" else Integer
            lower_bound = params["lower_bound"]
            upper_bound = params["upper_bound"]
            prior_distribution = params["prior"]
            search_space.append(
                class_(lower_bound, upper_bound, name=key, prior=prior_distribution)
            )
        elif params["type"] == "categorical":
            values = params["values"]
            prior_distribution = params["prior"]
            search_space.append(Categorical(values, name=key, prior=prior_distribution))
        else:
            raise ValueError("Hyperparameter type must be real or categorical.")
        x_init.append(params["init"])

    return search_space, x_init



def bayesian_search(X: pd.DataFrame, 
                    y: pd.Series, 
                    config: dict, 
                    num_points: int, 
                    optimize_on: tuple, 
                    log: bool = True) -> dict:


    loss_function = config['loss_function']
    metrics = config['metrics']
    cv_folds = config['cross_validation_folds']

    if log: 
        print('Starting Bayesian hyperparameter search...')
        print(f'Optimizing for {optimize_on[0]}...')

    search_space, x_init = configure_search_space()
        
    @use_named_args(dimensions=search_space)
    def objective_function(**hyperparameters) -> float:
        
        _, _, val_metrics = train(X=X,
                                  y=y,
                                  hyperparameters=hyperparameters,
                                  loss_function=loss_function,
                                  metrics=metrics,
                                  cross_val_folds=cv_folds,
        )
        best_metric = val_metrics[optimize_on[0]]

        if optimize_on[1] == 'minimize':
            return best_metric
        else: 
            return 1 - best_metric


    #Perform optimization
    num_initial_points = int(num_points) * 0.2
    num_initial_points = 3 if num_initial_points < 3 else num_initial_points

    results = gp_minimize(objective_function,
                          search_space,
                          n_calls=num_points,
                          n_initial_points=num_initial_points,
                          x0=x_init)

    best_hyperparameters = {str(var.name): results.x[i] for i, var in enumerate(search_space)}
    
    if log:
        print(f'Hyperparameters found with optimal {[optimize_on[0]]} : {1 - results.fun}')
        print(f'\t{best_hyperparameters}')
        
    return best_hyperparameters


