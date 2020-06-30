import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def one_hot_encode(df, column, prefix):
    return pd.get_dummies(df, columns=[column], prefix=prefix)

def prep_for_seq_lstm(df, query, timesteps, column='apn', prefix='target'):
    df = df.query(query)
    num_rows = df[column].value_counts().min() // timesteps * timesteps
    return one_hot_encode(df.groupby(column).head(num_rows), column, prefix)

def reshape_for_lstm(df, timesteps, df_name):
    print(df_name, end = ": ")
    df = df.reshape((df.shape[0], timesteps, df.shape[1] // timesteps))
    print(df.shape)
    return df

def get_energy_df(energy_sim, energy_actual, one_hot=True):
    actual_apn = energy_actual['apn'].unique()
    energy_sim = energy_sim[energy_sim['apn'].isin(actual_apn)]
    energy_actual = energy_actual.filter(items=['apn', 'year', 'month', 'day', 'hour', 'kwh'])
    energy_sim = energy_sim.filter(items=['apn', 'year', 'month', 'day', 'hour', 'kwh'])
    energy_actual.rename(columns={'kwh': 'kwh_actual_hold'}, inplace=True)
    energy_actual.sort_values(by=['apn', 'year', 'month', 'day', 'hour'], inplace=True)

    energy_sim['date'] = energy_sim['year'].astype(str) + "-" + energy_sim['month'].astype(str) +  "-" + energy_sim['day'].astype(str) + "-" + energy_sim['hour'].astype(str)
    energy_sim = energy_sim.pivot(index='date', columns='apn', values='kwh').add_prefix("kwh_sim_")
    energy_sim.reset_index(inplace=True)
    time_vars = energy_sim['date'].apply(lambda x: pd.to_numeric(pd.Series(x.split('-'))))
    time_vars.columns = ['year', 'month', 'day', 'hour']
    energy_sim = pd.concat([time_vars, energy_sim], axis=1)
    energy_sim.drop(columns='date', inplace=True)

    energy = energy_actual.merge(energy_sim, how='left')
    energy.sort_values(by=['apn', 'year', 'month', 'day', 'hour'], inplace=True)
    if one_hot:
        energy = one_hot_encode(energy, 'apn', 'target')
    energy['kwh_actual'] = energy['kwh_actual_hold']
    energy.drop(columns='kwh_actual_hold', inplace=True)
    
    return energy

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, remove_target=True, target_name='kwh_actual'):
    n_vars_correction = -1 if remove_target else 0
    n_vars = 1 if type(data) is list else data.shape[1] + n_vars_correction
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        if not remove_target:
            cols.append(dff.shift(i))
        else:
            cols.append(dff.shift(i).drop(columns=target_name))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        if i == 0:
            if remove_target:
                cols.append(dff.shift(-i))
            else:
                cols.append(dff.iloc[:, -1])
        else:
            cols.append(dff.shift(-i).drop(columns=target_name))
        if i == 0:
            if remove_target:
                n_vars = 1 if type(data) is list else data.shape[1]
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t)' % (n_vars+1))]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def preprocess(df, query, scaler, n_in, df_name, remove_target=True, lstm=True, target_name='kwh_actual', to_supervised=True):
    print(df_name, end = ": ")
    series = df
    if query is not None:
        series = df.query(query)
    
    series_x, series_y = pd.DataFrame(scaler.transform(series.drop(columns=target_name))), series.filter(items=[target_name])
    series_x.reset_index(drop=True, inplace=True)
    series_y.reset_index(drop=True, inplace=True)
    series_y = series_y.apply(lambda x: np.where(x < 1, 1, x)) # replace values less than 1 with 1
    
    x, y = None, None
    supervised = None
    
    seq_length = n_in
    if remove_target:
        seq_length = n_in + 1
    
    if to_supervised:
        supervised = series_to_supervised(pd.concat([series_x, series_y], axis=1), n_in=n_in, remove_target=remove_target, target_name=target_name)
        x, y = np.array(supervised.iloc[:,:-1]), np.array(supervised.iloc[:,-1])
    else:
        if lstm:
            sequence = np.array(pd.concat([series_x, series_y], axis=1))
            sequence = sequence.reshape((-1, n_in, sequence.shape[1]))
            x, y = sequence[:, :, :-1], np.expand_dims(sequence[:, :, -1], axis=-1)
            print(x.shape, y.shape)
            return x, y
        else:
            sequence = pd.concat([series_x, series_y], axis=1)
            x, y = np.array(sequence.iloc[:, :-1]), np.array(sequence.iloc[:, -1])
            print(x.shape, y.shape)
            return x, y
        
    if not lstm:
        print(x.shape, y.shape)
        return x, y
        
    x = reshape_for_lstm(seq_length)
    
    print(x.shape, y.shape)
    return x, y

def get_standard_scaler(df, query, target):
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(df.query(query).drop(columns=target))
    return standard_scaler

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
    
def print_metrics(model, x, y):
    y = y.ravel()
    y_predictions = model.predict(x).ravel()
    print("MAPE: " + str(100 * np.mean(np.abs(y_predictions / y - 1))))
    print("CV(RMSE): " + str(100 * np.sqrt(np.mean(np.power(y_predictions - y, 2))) / np.mean(y)))
    print("MBE: " + str(np.mean(y_predictions - y)))
    
def show_results(history, model, val_x, val_y, model_name = "Model"):
    plot_train_history(history, model_name + " Training and Validation Loss")
    print_metrics(model, val_x, val_y)