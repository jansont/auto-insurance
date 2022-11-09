import catboost as cb
import pandas as pd
import utils
from sklearn.model_selection import StratifiedKFold



def test(model: cb.CatBoostClassifier,
         X_test: pd.DataFrame,
         y_test: pd.Series, metrics: list[str]) -> dict:
    """
    Tests model and returns mean metrics as dict.
    """
    categorical_features_indices_test = utils.get_categorical_features(X_test)
    test_dataset = cb.Pool(X_test, y_test, cat_features=categorical_features_indices_test)
    test_metrics = model.eval_metrics(test_dataset, metrics)
    return test_metrics



def fit_and_validate(
    train_dataset: cb.Pool,
    hyperparameters: dict,
    val_dataset: cb.Pool = None,
    loss_function: str = "RMSE",
    metrics: list[str] = ["MAPE"],
    log: bool = False):

    """
    Training loop. Handles validation and cross validation.
    Args:x
        train_dataset : Prepared training set
        val_dataset : Prepared validation set
        Hyperparameters : Dictionary of hyperparameters
        loss function : specified as string (RMSE, MAPE, MAE)
        metrics : list of metrics to calculate
    Returns:
        model : trained CatBoost model
        train_metrics : dict of training metrics
        validation_metrics : dict of validation metrics.
    """
    validation_metrics, train_metrics = None, None
    # initialize models with optimized hyperparameters
    verbose = 1 if log else False
    # initialize models with optimized hyperparameters

    model = cb.CatBoostClassifier(loss_function=loss_function,
                                  verbose=verbose,
                                  **hyperparameters)

    use_best_model = False if val_dataset is None else True
    early_stopping = None if val_dataset is None else 10

    model.fit(
        X=train_dataset,  # CB.Pool
        use_best_model=use_best_model,  # early stopping
        eval_set=val_dataset,
        early_stopping_rounds=early_stopping,  # early stopping done w.r.t val set
    )

    # get best iteration before overfitting occurs
    best_iter = model.get_best_iteration()

    if val_dataset is not None:
        # evaluate performance on training and validation sets
        train_metrics = model.eval_metrics(train_dataset, metrics)
        validation_metrics = model.eval_metrics(val_dataset, metrics)
        # get performance on best iteration
        train_metrics = {name: m[best_iter] for name, m in train_metrics.items()}
        validation_metrics = {
            name: m[best_iter] for name, m in validation_metrics.items()
        }

    return model, train_metrics, validation_metrics




def train(X: pd.DataFrame,
          y: pd.Series,
          hyperparameters: dict,
          loss_function: str,
          metrics: list[str],
          cross_val_folds: int = 3, 
          validation: bool = True, 
          log = True):
    """
    Train model. Prepares data. Fits and validates model. Called by hyperparamter
    tuning or by training main.
    Args:
        X : training set
        y : regression labels
        hyperparamters: hyperparameter kwargs
        metrics : list of metrics (ie: [MAE, MAPE])
        test_size : proportion of test set to total set
        cross_val_folds : number of folds for cross val.
        validation : True to perform validation in training set.
        training director : directory in which train/test performance is tracked
    Returns:
        model : trained CatBoost regressor
        train metrics: dict of training performance
        validation_metrics: dict of validation performance
    """
    if not validation:
        categorical_features_indices_train = utils.get_categorical_features(X)
        train_dataset = cb.Pool(X, y, cat_features=categorical_features_indices_train)
        # val metrics is none

        if log: 
            print('Starting training (no validation) ...')

        model, train_metrics, val_metrics = fit_and_validate(train_dataset=train_dataset,
                                                            hyperparameters=hyperparameters,
                                                            val_dataset=None,
                                                            loss_function=loss_function, 
                                                            metrics=metrics)

        if log: 
            print('Done training.')
        
        
    #K fold cross validation
    else:
        if log: 
            print(f'Starting {cross_val_folds}-fold cross validation...')

        train_metrics_hist, val_metrics_hist = list(), list()
        skf = StratifiedKFold(n_splits=cross_val_folds, shuffle=True)
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            categorical_features_indices_train = utils.get_categorical_features(X_train)
            categorical_features_indices_val = utils.get_categorical_features(X_val)

            # Set up dataset for CatBoost
            train_dataset = cb.Pool(
                X_train, y_train, cat_features=categorical_features_indices_train
            )

            val_dataset = cb.Pool(
                X_val, y_val, cat_features=categorical_features_indices_val
            )

            # Train and evaluate CV fold
            model, train_metrics, val_metrics = fit_and_validate(train_dataset=train_dataset,
                                                                hyperparameters=hyperparameters,
                                                                val_dataset=val_dataset,
                                                                loss_function=loss_function, 
                                                                metrics=metrics)
            # Save results
            train_metrics_hist.append(train_metrics)
            val_metrics_hist.append(val_metrics)


        # get mean result across cross validation folds
        def dict_mean(dict_list: list[dict]) -> dict:
            mean_dict = {}
            for key in dict_list[0].keys():
                mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
            return mean_dict

        train_metrics = dict_mean(train_metrics_hist)
        val_metrics = dict_mean(val_metrics_hist)

        if log: 
            print('Done cross validation. Validation metrics:')
            v = {k:f'{v:.2f}' for k,v in val_metrics.items()}
            print(f'\t{v}')

    return model, train_metrics, val_metrics




def training_wrapper(train_path,
                     config, 
                     hyperparameters,
                     log = True) -> cb.CatBoostClassifier:
    """
    Entry point for training process (Not hyperparameter tuning).
    Trains model and tests and logs performance.
    Saves tested model.
    """

    # Read data and parameters
    X_train, y_train = utils.load_data(train_path)
    

    # parse parameters
    loss_function = config["loss_function"]
    metrics = config["metrics"]
    cross_val_folds = config["cross_validation_folds"]

    model, train_metrics, validation_metrics = train(X_train,
                                                    y_train,
                                                    hyperparameters,
                                                    loss_function, 
                                                    metrics, 
                                                    cross_val_folds, 
                                                    validation=True)
    
    if log: 
        print('Starting evaluation on test set...')


    # get best iteration and corresponding performance (early stopping)
    best_iter = model.get_best_iteration()


    results = {
        "Training": train_metrics,
        "Validation": validation_metrics,
        "Best_iteration": best_iter
    }
    
    if log: 
        print('Training and testing finished.')
        print(f'Re-training on entire training set till best iteration {best_iter}')
    
    hyperparameters['iterations'] = best_iter
    model, _, _ = train(X_train,
                        y_train,
                        hyperparameters,
                        loss_function, 
                        metrics, 
                        cross_val_folds, 
                        validation=False)


    return model, results