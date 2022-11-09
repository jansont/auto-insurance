from pathlib import Path
import catboost as cb
from utils import *


def predict(test_path: Path, model: cb.CatBoostClassifier, save:bool = True) -> list:

    # get features (and feature order) and target used to train model
    X,_ = load_data(test_path, test = True)

    # load evaluation set
  
    dataset = cb.Pool(
        X, cat_features=get_categorical_features(X)
    )

    predictions = model.predict(dataset)

    if save: 
        df = pd.read_csv(test_path)
        df['TARGET_FLAG'] = predictions
        df.to_csv('test_auto_predicted.csv')

    return predictions