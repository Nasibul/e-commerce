import pandas as pd
import pickle
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

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
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, min_delta=0.05, restore_best_weights=True)
es_scale = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, min_delta=0.0000005, restore_best_weights=True)
exog = pickle.load(
    open('/home/nasibul/Desktop/e-commerce/Advanced ARIMA/Data/exog.pkl', 'rb'))
exog['Guayaquil'][1297] = 1
for i in exog.columns:
    if all(exog[i].isin([0, 1])) and exog[i].nunique()==2:
        exog[i] = exog[i].astype('uint8')
log_dir = "logs/fit/" + dt.now().strftime('%m/%d/%Y %-I:%M:%S %p')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
learning_rates = [0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075,
                  0.0005, 0.00025, 0.0001, 0.000075, 0.00005, 0.000025, 0.00001]
time_step = 6

def data_prep(df, time_step=time_step, forecast=False, scale=False):
    if scale==True:
        indicator_variables_columns = []
        for i in df.columns:
            if df[i].dtype == 'uint8':
                indicator_variables_columns.append(i)
        indicator_variables = df[indicator_variables_columns]
        df = df.drop(indicator_variables_columns, axis=1)
        indicator_x = []
        for i in range(len(indicator_variables)-time_step):
            indicator_x.append(indicator_variables.iloc[i:i+time_step, :])
        indicator_x = np.array(indicator_x)


    X, y = [], []
    if forecast == False:
        for i in range(len(df)-time_step):
            X.append(df.iloc[i:i+time_step, :])
            y.append(df.iloc[i+time_step, :])
        X = np.array(X)
        y = np.array(y)
    else:
        forecast_length = 16
        for i in range(len(df)-time_step-forecast_length+1):
            X.append(df.iloc[i:i+time_step, :])
            y.append(df.iloc[i+time_step:i+time_step+forecast_length, :])
        X = np.array(X)
        y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=5, shuffle=False)
    if forecast == False:
        y_test = pd.DataFrame(y_test, columns=df.columns, index=df.index[-y_test.shape[0]:])
    else:
        y_test = train.iloc[-len(train)//10:, :]
    if scale==True:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_train = X_train.reshape(-1, X_train.shape[-1])
        X_test = X_test.reshape(-1, X_test.shape[-1])
        y_train = y_train.reshape(-1, y_train.shape[-1])
        if indicator_x.any():
            indicator_x = indicator_x.reshape(-1, indicator_x.shape[-1])
        X_train = scaler_x.fit_transform(X_train)
        if indicator_x.any():
            X_train = np.concatenate([indicator_x[:X_train.shape[0]], X_train], axis=1)
        X_test = scaler_x.transform(X_test)
        if indicator_x.any():
            X_test = np.concatenate([indicator_x[-X_test.shape[0]:], X_test], axis=1)
        y_train = scaler_y.fit_transform(y_train)
        X_train = X_train.reshape(-1, time_step, X_train.shape[-1])
        X_test = X_test.reshape(-1, time_step, X_test.shape[-1])        
        if forecast==True:
            y_train = y_train.reshape(-1, forecast_length, y_train.shape[-1])
        else:
            y_train = y_train.reshape(-1, y_train.shape[-1])
        data_sets = [X_train, X_test, y_train, y_test]
        for i in data_sets:
            print(i.shape)
        return X_train, X_test, y_train, y_test, scaler_y
    data_sets = [X_train, X_test, y_train, y_test]
    for i in data_sets:
            print(i.shape)
    return X_train, X_test, y_train, y_test


def predict(model, X_train, y_train, X_test, y_test, forecast=False, scaler_y=None, batch_size=32):
    x_val = X_train[1400:]
    y_val = y_train[1400:]
    if len(model.inputs)>1:
        x_val = []
        new_train = []
        for i in X_train:
            new_train.append(i[:1400])
            x_val.append(i[1400:])
        X_train = new_train
    if scaler_y==None:
        model.fit(X_train[:1400], y_train[:1400], epochs=1000, batch_size=batch_size, verbose=0, callbacks=[es, tensorboard], validation_data=(x_val, y_val))
        pred_model = model.predict(X_test)
        if forecast==False:
            pred = pd.DataFrame(pred_model, columns=y_test.columns, index=y_test.index)           
        else:
            pred = pd.DataFrame(pred_model[-1], columns=y_test.columns, index=test.index.unique())
    else:
        model.fit(X_train[:1400], y_train[:1400], epochs=1000, batch_size=batch_size, verbose=0, callbacks=[es_scale, tensorboard], validation_data=(x_val, y_val))
        pred_model = model.predict(X_test)
        if forecast==False:
            pred_model = scaler_y.inverse_transform(pred_model)
            pred = pd.DataFrame(pred_model, columns=y_test.columns, index=y_test.index)
        else:
            pred_model = scaler_y.inverse_transform(pred_model[-1])
            pred = pd.DataFrame(pred_model, columns=y_test.columns, index=test.index.unique())
    pred = abs(pred.iloc[:, -1782:])
    return pred


def compare(y_test, pred):
    if y_test.shape != pred.shape:
        print('Dimensions off')
    elif pred.isnull().values.any():
        print(f'{pred.isnull().sum()} null values')
    else:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            5, 1, sharex=True, layout='constrained')
        fig.set_size_inches(25, 10)
        fig.suptitle('Prediction vs Actual', fontsize=30)
        for i, ax in enumerate(fig.axes):
            df_test = y_test.iloc[:, i]
            prediction = pred.iloc[:, i]
            df_test.plot(ax=ax)
            prediction.plot(ax=ax)
            ax.set_xlabel('Date', fontsize=15)
            ax.set_ylabel('Amount', fontsize=15)
            ax.xaxis.set_tick_params(labelsize=15)
            ax.yaxis.set_tick_params(labelsize=15)
            ax.text(
                0.86, 0.75, f"R2 Score: {r2_score(df_test, prediction)}\nRMSLE: {mean_squared_log_error(df_test, prediction, squared=False)}", transform=ax.transAxes, fontsize=14)
        fig.legend([df_test, prediction],labels=['df_test', 'pred'], loc=(0.4, 0), fontsize=17)
        plt.show()


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


def tuning(model, X_train_mini, y_train_mini, X_val_mini, y_val_mini, scaler_y):
    tuner = keras_tuner.RandomSearch(
        model, objective='val_loss', max_trials=200, overwrite=True, max_consecutive_failed_trials=None)
    if scaler_y == None:
        tuner.search(X_train_mini, y_train_mini, epochs=20,
                    validation_data=(X_val_mini, y_val_mini), callbacks=[es])
    else:
        tuner.search(X_train_mini, y_train_mini, epochs=20,
                    validation_data=(X_val_mini, y_val_mini), callbacks=[es_scale])
    best_params = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_params)
    return model, best_params


def compare_all_series(y_test, pred):
    if y_test.shape != pred.shape:
        print('Dimensions off')
    elif pred.isnull().values.any():
        print(f'{pred.isnull().sum()} null values')
    else:
        R2 = []
        RMSLE = []
        for i in range(pred.shape[1]):
            df_test = y_test.iloc[:, i]
            prediction = pred.iloc[:, i]
            R2.append(r2_score(df_test, prediction))
            RMSLE.append(mean_squared_log_error(
                df_test, prediction, squared=False))
        results = pd.DataFrame({'Store_product': y_test.columns,
                                'R2 Score': R2,
                                'Root Mean Squared Log Error': RMSLE})
        print(results.mean())
        return results

def custom_compare(y_test, pred, cols_list):
    fig, axes = plt.subplots(len(cols_list), 1, sharex=True, layout='constrained')
    fig.set_size_inches(25, 3*len(cols_list))
    fig.suptitle('Prediction vs Actual', fontsize=30)
    column_names = [y_test.columns[num-1] for num in cols_list]
    for i, ax in zip(column_names, axes):
        df_test = y_test.loc[:, [i]]
        prediction = pred.loc[:, [i]]
        df_test.plot(ax=ax, legend=None)
        prediction.plot(ax=ax, legend=None)
        ax.set_xlabel('Date', fontsize=15)
        ax.set_ylabel('Amount', fontsize=15)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.text(0.86, 0.74, f"{i}\nR2 Score: {r2_score(df_test, prediction)}\nRMSLE: {mean_squared_log_error(df_test, prediction, squared=False)}", transform=ax.transAxes, fontsize=14)
    fig.legend([df_test, prediction],labels=['df_test', 'pred'], loc=(0.4, 0), fontsize=17)
    plt.show()


def experiment(X_train, X_test, y_train, y_test, model, model_number, description, tune=True, forecast=False, scaler_y=None, batch_size=32):
    if tune==True:
        y_train_mini = y_train[:200, :]
        y_val_mini = y_train[200:220, :]
        if 'output' in model.__code__.co_varnames:
            X_train_mini = []
            X_val_mini = []
            for i in X_train:
                X_train_mini.append(i[:200, :])
                X_val_mini.append(i[200:220, :])
        else:
            X_train_mini = X_train[:200, :]
            X_val_mini = X_train[200:220, :]
        hypertuned_model, best_params = tuning(model, X_train_mini, y_train_mini, X_val_mini, y_val_mini, scaler_y)
    else:
        hp = keras_tuner.HyperParameters()
        hp.Fixed('learning_rate', 0.00075)
        hp.Fixed('optimizer', 'adam')
        best_params = 'No tuning'
        hypertuned_model = model(hp)
    pred = predict(hypertuned_model, X_train, y_train, X_test, y_test, forecast=forecast, scaler_y=scaler_y, batch_size=batch_size)
    compare(y_test.iloc[:, -1782:], pred)
    results = compare_all_series(y_test.iloc[:, -1782:], pred)
    full_experiment = {'model_number': model_number,
                       'params': best_params,
                       'Description': description,
                       'Original test set': y_test.iloc[:, -1782:],
                       'Predictions': pred, 
                       'Results': results,
                       'Aggregated Results': results.mean()}
    pickle.dump(full_experiment, open(f'/home/nasibul/Desktop/e-commerce/Advanced ARIMA/Data/{model_number}.pkl', 'wb'))
    
    
