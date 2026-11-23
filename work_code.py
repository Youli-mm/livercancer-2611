import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import lasagne
import deepsurv
from deepsurv import deep_surv
from deepsurv.deepsurv_logger import DeepSurvLogger, TensorboardLogger
import utils
train_dataset_fp = './train_data.csv'
train_df = pd.read_csv(train_dataset_fp)
def dataframe_to_deepsurv_ds(df, event_col ='Event', time_col = 'Time'):
    e = df[event_col].values.astype(np.int32)
    t = df[time_col].values.astype(np.float32)
    x_df = df.drop([event_col, time_col], axis = 1)
    x = x_df.values.astype(np.float32)
    return {
        'x' : x,
        'e' : e,
        't' : t
    }
train_data = dataframe_to_deepsurv_ds(train_df, event_col = 'Event', time_col= 'Time')
hyperparams = {
    'L2_reg': 10.0,
    'batch_norm': True,
    'dropout': 0.4,
    'hidden_layers_sizes': [25, 25],
    'learning_rate': 1e-05,
    'lr_decay': 0.001,
    'momentum': 0.9,
    'n_in': train_data['x'].shape[1],
    'standardize': True
}
model = deep_surv.DeepSurv(**hyperparams)
experiment_name = 'test_experiment_sebastian'
logdir = './logs/tensorboard/'
logger = TensorboardLogger(experiment_name, logdir=logdir)
update_fn=lasagne.updates.nesterov_momentum
metrics = model.train(train_data, n_epochs=500, logger=logger, update_fn=update_fn)
print('Train C-Index:', metrics['c-index'][-1])
deepsurv.plot_log(metrics)
risk_train=model.predict_risk(train_data["x"])