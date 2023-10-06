import numpy as np
from sklearn import metrics

def print_metrics(y_test,y_pred):
    print('MAE:', metrics.mean_absolute_error(np.exp(y_test), np.exp(y_pred)))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(np.exp(y_test), np.exp(y_pred))))
    print('R2:',  metrics.r2_score(np.exp(y_test), np.exp(y_pred)))
    print('MAPE:', mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred)))
    pass

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def dataframe_metrics(y_test,y_pred):
    stats = [
       metrics.mean_absolute_error(np.exp(y_test), np.exp(y_pred)),
       np.sqrt(metrics.mean_squared_error(np.exp(y_test), np.exp(y_pred))),
       metrics.r2_score(np.exp(y_test), np.exp(y_pred)),
       mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred))
    ]
    return stats
# measured_metrics = pd.DataFrame({"error_type":["MAE", "RMSE", "R2", "MAPE"]})
# measured_metrics.set_index("error_type")