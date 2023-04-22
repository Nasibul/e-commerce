import pandas as pd
import pickle
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
import keras_tuner

train = pd.read_csv(
    '/home/nasibul/Desktop/e-commerce/Advanced ARIMA/Data/train.csv', index_col='date')
train.index = pd.to_datetime(train.index)
train = pd.pivot_table(train, values='sales', index="date", columns=[
                       'store_nbr', 'family'], aggfunc=np.sum)
train.columns = [f"{a[0]}_{a[1].replace('/','_')}" for a in train.columns]
test = pd.read_csv(
    '/home/nasibul/Desktop/e-commerce/Advanced ARIMA/Data/test.csv', index_col='date')
test.index = pd.to_datetime(test.index)
es = keras.callbacks.EarlyStopping(
    monitor='loss', mode='min', patience=10, min_delta=0.05)
exog = pickle.load(
    open('/home/nasibul/Desktop/e-commerce/Advanced ARIMA/Data/exog.pkl', 'rb'))
log_dir = "logs/fit/" + dt.now().strftime('%m/%d/%Y %-I:%M:%S %p')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
learning_rates = [0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075,
                  0.0005, 0.00025, 0.0001, 0.000075, 0.00005, 0.000025, 0.00001]


def data_prep(X, Y, scale=False):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=5, shuffle=False)
    X_test_copy = X_test
    X_test_index = X_test.index
    if scale == False:
        X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_train = y_train.values.reshape(y_train.shape[0], y_train.shape[1], 1)
        y_test = y_test.values.reshape(y_test.shape[0], y_test.shape[1], 1)
    else:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train = scaler.fit_transform(y_train)
        y_test = scaler.transform(y_test)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
        return X_train, X_test, y_train, y_test, X_test_copy, X_test_index, scaler
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test, X_test_copy, X_test_index


def predict(model, X_train, y_train, X_test, X_test_copy, X_test_index, forecast=False, scaler=None):
    if scaler==None:
        model.fit(X_train, y_train, epochs=500, verbose=0, callbacks=[es, tensorboard])
        pred_model = model.predict(X_test)
    else:
        es_scale = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=10, min_delta=0.005)
        model.fit(X_train, y_train, epochs=500, verbose=0, callbacks=[es_scale, tensorboard])
        pred_model = model.predict(X_test)
    if forecast == False:
        pred = pred_model.reshape(X_test_copy.shape[0], X_test_copy.shape[1])
        pred = pd.DataFrame(pred, columns=X_test_copy.columns, index=X_test_index)
        pred = pred.iloc[:, -1782:]
    else:
        pred_model = scaler.inverse_transform(pred_model[168].T)
        pred_model = pred_model.reshape(16, X_test_copy.shape[1])
        pred = pd.DataFrame(pred_model, columns=train.columns, index=test.index.unique())
        pred = pred.iloc[:, -1782:]
    return pred


def compare(X_test_copy, pred):
    if X_test_copy.shape != pred.shape:
        print('Dimensions off')
    elif pred.isnull().values.any():
        print(f'{pred.isnull().sum()} null values')
    else:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            5, 1, sharex=True, layout='constrained')
        fig.set_size_inches(25, 10)
        fig.suptitle('Prediction vs Actual', fontsize=30)
        for i, ax in enumerate(fig.axes):
            df_test = X_test_copy.iloc[:, i]
            prediction = pred.iloc[:, i]
            df_test.plot(ax=ax)
            prediction.plot(ax=ax)
            ax.set_xlabel('Date', fontsize=15)
            ax.set_ylabel('Amount', fontsize=15)
            ax.xaxis.set_tick_params(labelsize=15)
            ax.yaxis.set_tick_params(labelsize=15)
            ax.text(
                0.86, 0.75, f"R2 Score: {r2_score(df_test, prediction)}\nRMSLE: {mean_squared_log_error(df_test, prediction, squared=False)}", transform=ax.transAxes, fontsize=14)
        plt.show()


def Y_forecast(X):
    Y = pd.DataFrame()
    for feature in X.columns:
        for step in range(1, 17):
            Y[f'{feature}_step_ahead_{step}'] = X[feature].shift(
                -step, fill_value=0)
    return Y


def submit(pred):
    id = []
    sales = []
    for i in test.id:
        row = test[test['id'] == i]
        column = f"{row['store_nbr'][0]}_{row['family'][0].replace('/', '_')}"
        value = pred[column][pred.index == row.index[0]][0]
        # The info from the ID row is taken and used to form a variable called column.
        # This column variable refers to the correct store in the dataframe and the equality check on the indexes refer to the proper dates lining up.
        id.append(i)
        sales.append(value)
    submission = pd.DataFrame({'id': id, 'sales': sales})
    print(submission.head(10))
    return submission


def tuning(model, X_train_mini, y_train_mini, X_val_mini, y_val_mini):
    tuner = keras_tuner.RandomSearch(
        model, objective='val_loss', max_trials=60, overwrite=True, max_consecutive_failed_trials=None)
    tuner.search(X_train_mini, y_train_mini, epochs=10,
                 validation_data=(X_val_mini, y_val_mini))
    best_params = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_params)
    return model, best_params


def compare_all_series(X_test_copy, pred):
    if X_test_copy.shape != pred.shape:
        print('Dimensions off')
    elif pred.isnull().values.any():
        print(f'{pred.isnull().sum()} null values')
    else:
        R2 = []
        RMSLE = []
        for i in range(pred.shape[1]):
            df_test = X_test_copy.iloc[:, i]
            prediction = pred.iloc[:, i]
            R2.append(r2_score(df_test, prediction))
            RMSLE.append(mean_squared_log_error(
                df_test, prediction, squared=False))
        results = pd.DataFrame({'Store_product': X_test_copy.columns,
                                'R2 Score': R2,
                                'Root Mean Squared Log Error': RMSLE})
        print(results.mean())
        return results


def experiment(X_train, X_test, y_train, y_test, X_test_copy, X_test_index, model, model_number, description, tune=True, forecast=False, scaler=None):
    if tune==True:
        y_train_mini = y_train[:150, :]
        y_val_mini = y_train[151:160, :]
        if 'output' in model.__code__.co_varnames:
            X_train_mini = []
            X_val_mini = []
            for i in X_train:
                X_train_mini.append(i[:150, :])
                X_val_mini.append(i[151:160, :])
        else:
            X_train_mini = X_train[:150, :]
            X_val_mini = X_train[151:160, :]
        hypertuned_model, best_params = tuning(model, X_train_mini, y_train_mini, X_val_mini, y_val_mini)
    else:
        hp = keras_tuner.HyperParameters()
        hp.Fixed('learning_rate', 0.001)
        hp.Fixed('optimizer', 'adam')
        best_params = 'No tuning'
        hypertuned_model = model(hp)
    pred = predict(hypertuned_model, X_train, y_train, X_test, X_test_copy, X_test_index, forecast=forecast, scaler=scaler)
    compare(X_test_copy.iloc[:, -1782:], pred)
    results = compare_all_series(X_test_copy.iloc[:, -1782:], pred)
    full_experiment = {'model_number': model_number,
                       'params': best_params,
                       'Description': description,
                       'Results': results,
                       'Aggregated Results': results.mean()}
    pickle.dump(full_experiment, open(f'/home/nasibul/Desktop/e-commerce/Advanced ARIMA/Data/{model_number}.pkl', 'wb'))
    
    
