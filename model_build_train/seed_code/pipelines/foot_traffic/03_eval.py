
import json
import pathlib
import pickle
import tarfile

import joblib
import numpy as np
import pandas as pd
import xgboost
import xgboost as xgb
from typing import Tuple, Dict

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import os

def load_data(test_path:str) -> Tuple[xgb.DMatrix, np.array]:
    
    df = pd.read_csv(test_path, header=None)
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    
    X_test = xgboost.DMatrix(df.values)
    
    return X_test, y_test, df


def evaluate_model(
    model_path:str, X_test: xgb.DMatrix, y_test: np.array
)-> Dict[str, any]:
    """Calculates and logs the coefficient of determination.

    Args:
        model_path: Trained model path.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
        
    model = pickle.load(open("xgboost-model", "rb"))
    
    preds = model.predict(X_test)
    
    # metrics:
    mse = mean_squared_error(y_test, preds)
    std = np.std(y_test - preds)
    r2 = r2_score(y_test, preds)
    print(r2)
    
    score_dict = {
        "regression_metrics": {
            "mse": {"value": mse, "standard_deviation": std},
       },
    }

    return score_dict


def generate_plots(model_path:str, X_test: xgb.DMatrix, y_test: np.array):
    
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
        
    model = pickle.load(open("xgboost-model", "rb"))
    
    preds = model.predict(X_test)
    
    
    plot_pred = plt.scatter(y_test, preds, color='b', alpha = 0.6)
    plt.title('Cases sold vs Predicted Cases')
    plt.xlabel('cases sold')
    plt.ylabel('predicted cases sold')
    
    pred_plot = plt
    return pred_plot


def compare_cases_vs_foot_traffic_plot(model_path:str, 
                                       X_test: xgb.DMatrix,
                                       y_test: np.array,
                                       df:pd.DataFrame):
    
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
        
    model = pickle.load(open("xgboost-model", "rb"))
    
    preds = model.predict(X_test)
    
    foot_traffic = df.iloc[:,-1].values
    
    plot_pred = plt.scatter(foot_traffic, preds, color='b', alpha = 0.6)
    plot_test = plt.scatter(foot_traffic, y_test, color='g', alpha = 0.6)
    plt.legend((plot_pred, plot_test), ('prediction', 'truth'), loc='upper left', fontsize=8)
    plt.title('Prediction of cases sold vs foot traffic, per week')
    plt.xlabel('Foot Traffic')
    plt.ylabel('Total cases sold')

    traffic_plot = plt
    return traffic_plot
    
    
    
    
if __name__ == "__main__":

    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_path = "/opt/ml/processing/test/test.csv"

    X_test, y_test, df = load_data(test_path)

    score_dict = evaluate_model(model_path, X_test, y_test)
    
    pred_plt = generate_plots(model_path, X_test, y_test)

    output_dir = "/opt/ml/processing/evaluation"
    plot_dir = "/opt/ml/processing/chart"

    #pathlib.Path(output_dir).mkdir(parents=True, exists_ok=True) # (reqiures python > 3.5)
    os.makedirs(output_dir, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(score_dict))
    
    
    
    plots_path = f"{plot_dir}/preds/preds_plot.png"
    
    pred_plt.savefig(plots_path, bbox_inches='tight')
    
    
    
    traffic_plt = compare_cases_vs_foot_traffic_plot(model_path, X_test, y_test, df)
    
    plot_vs_traffic_path = f"{plot_dir}/traffic/traffic_plot.png"
    
    traffic_plt.savefig(plot_vs_traffic_path, bbox_inches='tight')
    
    print("completed")
    
    