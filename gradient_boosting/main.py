from hyperparameters import *
from utils import *
from train import *
from inference import *

train_path = '../auto-insurance-fall-2017/train_auto.csv'
test_path = '../auto-insurance-fall-2017/test_auto.csv'

train_X, train_y = utils.load_data(train_path)

config = {
            'loss_function':'CrossEntropy', 
            'metrics': ['MCC', 'F1', 'AUC', 'Accuracy', 'Precision', 'Recall'],
            'cross_validation_folds' : 5
        }

optimize_on = ('F1', 'maximize')

hyperparameters = bayesian_search(train_X, 
                                  train_y, 
                                  config, 
                                  optimize_on = optimize_on,
                                  num_points = 100, 
                                  log = True)

model, results = training_wrapper(train_path,
                                  config, 
                                  hyperparameters, 
                                  log = True)

predictions = predict('../auto-insurance-fall-2017/test_auto.csv',
                     model,
                     save = True)
