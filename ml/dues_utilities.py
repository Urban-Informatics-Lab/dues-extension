import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def one_hot_encode(df, column, prefix):
    return pd.get_dummies(df, columns=[column], prefix=prefix)
    
def reshape_for_lstm(df, timesteps, df_name):
    print(df_name, end = ": ")
    df = df.reshape((df.shape[0], timesteps, df.shape[1] // timesteps))
    print(df.shape)
    return df

def equalize_timesteps(df, query, timesteps, column='apn', one_hot=True, prefix='target'):
    df = df.query(query)
    num_rows = df[column].value_counts().min() // timesteps * timesteps
    if one_hot:
        return one_hot_encode(df.groupby(column).head(num_rows), column, prefix)
    return df.groupby(column).head(num_rows)

def agg_temporal(df, by_apn, temporal_scale, func):
    if temporal_scale == 'hour':
        return df
    groupings = ['apn', 'year', 'month', 'day']
    if not by_apn:
        groupings = ['year', 'month', 'day']
    thresholds = {'month': 28, 'day': 24}
    df = df.drop(columns='hour')
    while(groupings[-1] != temporal_scale):
        df = df.groupby(groupings).filter(lambda x: len(x) >= thresholds[groupings[-1]]).groupby(groupings).agg('sum')
        del groupings[-1]
    return df.groupby(groupings).filter(lambda x: len(x) >= thresholds[groupings[-1]]).groupby(groupings).agg('sum').reset_index()
        
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
    if query is not None:
        standard_scaler.fit(df.query(query).drop(columns=target))
    else:
        standard_scaler.fit(df.drop(columns=target))
    return standard_scaler

def train_and_evaluate_model(model, train_x, train_y, val_x, val_y, test_x, test_y, optimizer, loss, batch_size, shuffle, epochs, callbacks, verbose, model_name="Model", print_model_summary = False, val_agg_df=None, test_agg_df=None):
    
    model.compile(optimizer=optimizer, loss=loss)
    
    if print_model_summary:
        model.summary()

    history = model.fit(
        train_x, train_y, 
        validation_data=[val_x, val_y], 
        batch_size = batch_size, 
        shuffle = shuffle, 
        epochs=epochs, 
        callbacks=callbacks,
        verbose=verbose
    )
    
    val_metrics = get_metrics_and_plot(history, model, val_x, val_y, model_name, print_results=False, agg_df=val_agg_df)
    test_metrics = get_metrics_and_plot(history, model, test_x, test_y, model_name, plot=False, print_results=False, agg_df=test_agg_df)
    
    return val_metrics, test_metrics

def train_and_evaluate_simple_model(model, train_x, train_y, val_x, val_y, test_x, test_y):
    model = model.fit(train_x, train_y)
    
    val_metrics = get_metrics(model, val_x, val_y, print_results=False)
    test_metrics = get_metrics(model, test_x, test_y, print_results=False)
    
    return val_metrics, test_metrics
    

def walk_forward_cv(model, train_folds, val_folds, test_folds, optimizer='adam', loss='mae', batch_size=32, shuffle=True, epochs=30, callbacks=None, verbose=0, simple=False, model_name = "Model", val_agg_df=None, test_agg_df=None):
    num_folds = len(train_folds)
    
    agg_val_metrics, agg_test_metrics = np.zeros((2, 3, 4)), np.zeros((2, 3, 4))
    if val_agg_df is None:
        agg_val_metrics, agg_test_metrics = np.zeros(4), np.zeros(4)
    
    for idx, train_fold in enumerate(train_folds):
        print("\nFold #" + str(idx + 1))
        val_fold = val_folds[idx]
        test_fold = test_folds[idx]
        
        if simple:
            val_metrics, test_metrics = train_and_evaluate_simple_model(
                model,
                train_fold[0], train_fold[1], 
                val_fold[0], val_fold[1], 
                test_fold[0], test_fold[1]
            )
        else:
            val_input = val_agg_df
            test_input = test_agg_df
            if val_agg_df is not None:
                val_input = val_agg_df[idx]
                test_input = test_agg_df[idx]
                
            val_metrics, test_metrics = train_and_evaluate_model(
                model,
                train_fold[0], train_fold[1], 
                val_fold[0], val_fold[1], 
                test_fold[0], test_fold[1],
                optimizer,
                loss,
                batch_size,
                shuffle,
                epochs,
                callbacks,
                verbose,
                model_name,
                print_model_summary = idx == 0,
                val_agg_df=val_input, test_agg_df=test_input
            )
               

        if val_agg_df is None:
            print("\nValidation:")
            print_metrics(*val_metrics)
            print("\nTest:")
            print_metrics(*test_metrics)
            
        # Remove if statement and uncomment code below for cross-fold averages.
        if idx == num_folds - 1:
            agg_val_metrics += np.array(val_metrics) # / num_folds
            agg_test_metrics += np.array(test_metrics) # / num_folds

#     if val_agg_df is None:
#         print("\nAll-Fold Average")
#         print("\nValidation:")
#         print_metrics(*agg_val_metrics)
#         print("\nTest:")
#         print_metrics(*agg_test_metrics)
    
    return np.stack([agg_val_metrics, agg_test_metrics])

def repeat_experiment(model, N, train_folds, val_folds, test_folds, callbacks, verbose, model_name="Model", val_agg_df=None, test_agg_df=None):
    untrained_model_weights = model.get_weights()
    
    metrics = np.zeros((2,4))
    if val_agg_df is not None:
        metrics = np.zeros((2, 2, 3, 4))

    for num_exp in range(N):
        exp_model = model
        exp_model.set_weights(untrained_model_weights)
        print("-----Experiment #" + str(num_exp + 1) + "-----")
        metrics = metrics + walk_forward_cv(
            exp_model, 
            train_folds, 
            val_folds, 
            test_folds, 
            callbacks=callbacks, 
            verbose=verbose, 
            model_name=model_name,
            val_agg_df=val_agg_df, test_agg_df=test_agg_df
        ) / N
        
    return metrics
    

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
    
def get_metrics(model, x, y, print_results=True, agg_df=None):
    y = y.ravel()
    y_predictions = model.predict(x).ravel()
 
    metrics = np.zeros((2, 3, 4))
    
    mape = 100 * np.mean(np.abs(y_predictions / y - 1))

    if agg_df is not None:
        metrics[0][0] = agg_metrics(agg_df, y_predictions, 'hour')
        metrics[0][1] = agg_metrics(agg_df, y_predictions, 'day')
        metrics[0][2] = agg_metrics(agg_df, y_predictions, 'month')
        metrics[1][0] = agg_metrics(agg_df, y_predictions, 'hour', 'urban')
        metrics[1][1] = agg_metrics(agg_df, y_predictions, 'day', 'urban')
        metrics[1][2] = agg_metrics(agg_df, y_predictions, 'month', 'urban')
        
        return metrics

    mape = 100 * np.mean(np.abs(y_predictions / y - 1))
    mse = np.mean(np.power(y_predictions - y, 2))
    cv_rmse = 100 * np.sqrt(np.mean(np.power(y_predictions - y, 2))) / np.mean(y)
    mbe = np.mean(y_predictions - y)
    
    if print_results:
        print_metrics(mape, mse, cv_rmse, mbe)
        
    return mape, mse, cv_rmse, mbe
    
def get_metrics_and_plot(history, model, x, y, model_name = "Model", plot=True, print_results=True, agg_df=None):
    if plot:
        plot_train_history(history, model_name + " Training and Validation Loss")
    return get_metrics(model, x, y, print_results, agg_df=agg_df)

def print_metrics(mape, mse, cv_rmse, mbe):
    print("MAPE: " + str(mape))
    print("MSE: " + str(mse))
    print("CV(RMSE): " + str(cv_rmse))
    print("MBE: " + str(mbe))

def agg_metrics(agg_df, y_predictions, temporal_scale, spatial_scale='building'):
    agg_df['kwh_pred'] = y_predictions
    
    energy_agg = agg_temporal(agg_df, by_apn=True, temporal_scale=temporal_scale, func='sum')
    if spatial_scale == "urban":
        energy_agg = energy_agg.groupby(energy_agg.columns[energy_agg.columns.isin(['year', 'month', 'day', 'hour'])].tolist()).agg('sum').reset_index()
    y_actual = energy_agg['kwh_actual']
    y_predictions = energy_agg['kwh_pred']
    
    mape = 100 * np.mean(np.abs(y_predictions / y_actual - 1))
    mse = np.mean(np.power(y_predictions - y_actual, 2))
    cv_rmse = 100 * np.sqrt(np.mean(np.power(y_predictions - y_actual, 2))) / np.mean(y_actual)
    mbe = np.mean(y_predictions - y_actual)
    
    return mape, mse, cv_rmse, mbe

def print_overall_metrics(metrics):
    print("\nOverall Validation:")
    print_metrics(*metrics[0])
    print("\nOverall Test:")
    print_metrics(*metrics[1])

def get_energy_df(energy_sim, energy_actual, one_hot=True, spatial_scale='building', temporal_scale='hour', add_context=True):
    print("Retrieving DUE-S Energy Data...")
    
    actual_apn = energy_actual['apn'].unique()
    energy_sim = energy_sim[energy_sim['apn'].isin(actual_apn)]
    energy_actual = energy_actual.filter(items=['apn', 'year', 'month', 'day', 'day_of_week', 'hour', 'kwh'])
    energy_sim = energy_sim.filter(items=['apn', 'year', 'month', 'day', 'hour', 'kwh'])
    energy_actual.rename(columns={'kwh': 'kwh_actual_hold'}, inplace=True)
    energy_actual.sort_values(by=['apn', 'year', 'month', 'day', 'hour'], inplace=True)
    
    groupings = ['year', 'month', 'day', 'hour']
    if spatial_scale == 'urban':
        NUM_BUILDINGS = len(actual_apn)
        energy_actual = energy_actual.groupby(groupings).filter(lambda x: len(x) == NUM_BUILDINGS)
        energy_actual = energy_actual.groupby(groupings).agg('sum').reset_index()
        energy_actual = agg_temporal(energy_actual, by_apn=False, temporal_scale=temporal_scale, func='sum')
    else:
        energy_actual = agg_temporal(energy_actual, by_apn=True, temporal_scale=temporal_scale, func='sum')

    if add_context:
        energy_sim['date'] = energy_sim['year'].astype(str) + "-" + energy_sim['month'].astype(str) +  "-" + energy_sim['day'].astype(str) + "-" + energy_sim['hour'].astype(str)
        energy_sim = energy_sim.pivot(index='date', columns='apn', values='kwh').add_prefix("kwh_sim_")
        energy_sim.reset_index(inplace=True)
        time_vars = energy_sim['date'].apply(lambda x: pd.to_numeric(pd.Series(x.split('-'))))
        time_vars.columns = ['year', 'month', 'day', 'hour']
        energy_sim = pd.concat([time_vars, energy_sim], axis=1)
        energy_sim.drop(columns='date', inplace=True)
    
    energy_sim = agg_temporal(energy_sim, by_apn=False, temporal_scale=temporal_scale, func='sum')

    energy = energy_actual.merge(energy_sim, how='left')
    energy = energy[~energy.isnull().any(axis=1)]
    
    if spatial_scale == 'building':
        groupings = ['apn', 'year', 'month', 'day', 'hour']
        t_index = groupings.index(temporal_scale) + 1
        grouping  = groupings[:t_index]
        energy.sort_values(by=grouping, inplace=True)
        if one_hot:
            energy = one_hot_encode(energy, 'apn', 'target')
    elif spatial_scale == 'urban':
        groupings = ['year', 'month', 'day', 'hour']
        t_index = groupings.index(temporal_scale) + 1
        grouping  = groupings[:t_index]
        energy.sort_values(by=grouping, inplace=True)
        
    energy['kwh_actual'] = energy['kwh_actual_hold']
    energy.drop(columns='kwh_actual_hold', inplace=True)

    return energy

def make_supervised_arrays(df, fold_query, timesteps, scaler, df_name):
    df = equalize_timesteps(df, fold_query, timesteps)
#     df = one_hot_encode(df, 'day_of_week', 'wday')
#     df.drop(columns=['year', 'day', 'hour'], inplace=True)
#     df_no_target = df.filter(regex='^(month|kwh|wday)', axis=1)
    df_no_target = df.filter(regex='^kwh', axis=1)
    df_processed = preprocess(df_no_target, None, scaler, n_in=timesteps - 1, df_name=df_name, lstm=False)
    df_x = np.hstack([df_processed[0], df.filter(regex='^target', axis=1).iloc[23:, :]])
    df_y = df_processed[1]
    
    return df_x, df_y

def get_folds(n):
    
    if n == 3:
        fold_1 = ['year <= 2016 and month < 7', 'year == 2016 and month >= 7', 'year == 2017 and month < 7']
        fold_2 = ['year <= 2016', 'year == 2017 and month < 7', 'year == 2017 and month >= 7']
        fold_3 = ['(year == 2017 and month < 7) or year <= 2016', 'year == 2017 and month >= 7', 'year == 2018']
        return [fold_1, fold_2, fold_3]
    elif n == 4:
        fold_1 = ['year <= 2016 and month < 7', 'year == 2016 and month >= 7', 'year == 2017 and month < 7']
        fold_2 = ['year <= 2016', 'year == 2017 and month < 7', 'year == 2017 and month >= 7']
        fold_3 = ['(year == 2017 and month < 7) or year <= 2016', 'year == 2017 and month >= 7', 'year == 2018 and month < 7']
        fold_4 = ['year <= 2017', 'year == 2018 and month < 7', 'year == 2018 and month >= 7']
        return [fold_1, fold_2, fold_3, fold_4]
    
    print("Error: Unsupported number of folds.")
    
def make_supervised_folds(folds, energy, timesteps, scaler):
    
    train_folds = []
    val_folds = []
    test_folds = []
    
    n_prev = timesteps - 1

    for idx, fold in enumerate(folds):
        print("Processing Fold #" + str(idx + 1) + "...")
        
        energy_train = equalize_timesteps(energy, fold[0], timesteps)
        #energy_train = one_hot_encode(energy_train, 'day_of_week', 'wday')
        #energy_train.drop(columns=['year', 'day', 'hour'], inplace=True)
        #train_no_target = energy_train.filter(regex='^(month|kwh|wday)', axis=1)
        train_no_target = energy_train.filter(regex='^kwh', axis=1)
        
        scaler = get_standard_scaler(train_no_target, None, 'kwh_actual')

        train_folds.append(make_supervised_arrays(energy, fold[0], timesteps, scaler, "Train"))
        val_folds.append(make_supervised_arrays(energy, fold[1], timesteps, scaler, "Val"))
        test_folds.append(make_supervised_arrays(energy, fold[2], timesteps, scaler, "Test"))
        
    return train_folds, val_folds, test_folds

def make_seq_folds(folds, energy, timesteps, scalers, lstm=True, get_scalers=False):

    train_folds = []
    val_folds = []
    test_folds = []
    
    scalers_return = []

    for idx, fold in enumerate(folds):
        print("Processing Fold #" + str(idx + 1) + "...")

        energy_train = equalize_timesteps(energy, fold[0], timesteps)      
#         energy_train = one_hot_encode(energy_train, 'day_of_week', 'wday')
#         energy_train.drop(columns=['year', 'day', 'hour'], inplace=True)
        energy_train.drop(columns=['year', 'day', 'hour', 'month', 'day_of_week'], inplace=True)
        
        if get_scalers:
            scalers_return.append(get_standard_scaler(energy_train, None, 'kwh_actual'))
            continue
        
        scaler = None
        if scalers is None:
            scaler = get_standard_scaler(energy_train, None, 'kwh_actual')
        else:
            scaler = scalers[idx]
            
        energy_val = equalize_timesteps(energy, fold[1], timesteps)
#         energy_val = one_hot_encode(energy_val, 'day_of_week', 'wday')
#         energy_val.drop(columns=['year', 'day', 'hour'], inplace=True)
        energy_val.drop(columns=['year', 'day', 'hour', 'month', 'day_of_week'], inplace=True)

        energy_test = equalize_timesteps(energy, fold[2], timesteps)
#         energy_test = one_hot_encode(energy_test, 'day_of_week', 'wday')
#         energy_test.drop(columns=['year', 'day', 'hour'], inplace=True)
        energy_test.drop(columns=['year', 'day', 'hour', 'month', 'day_of_week'], inplace=True)

        train_folds.append(preprocess(energy_train, None, scaler, n_in=timesteps, df_name="Train", remove_target=False, lstm=lstm, to_supervised=False))
        val_folds.append(preprocess(energy_val, None, scaler, n_in=timesteps, df_name="Val", remove_target=False, lstm=lstm, to_supervised=False))
        test_folds.append(preprocess(energy_test, None, scaler, n_in=timesteps, df_name="Test", remove_target=False, lstm=lstm, to_supervised=False))
        
    if get_scalers:
        return scalers_return
    
    return train_folds, val_folds, test_folds

def make_test_fold_fast(df, query, num_obs, target_one_hot, scaler, timesteps):
    df = df.query(query).groupby('apn').head(num_obs)
    df = pd.concat((df, target_one_hot), axis=1)
    
    series_x = pd.DataFrame(scaler.transform(df.drop(columns=['apn', 'year', 'day', 'hour', 'month', 'day_of_week', 'kwh_actual'])))
    series_x.reset_index(drop=True, inplace=True)
    
    sequence = np.array(series_x)
    
    return sequence.reshape((-1, timesteps, sequence.shape[1]))

def calc_savings(model, energy, energy_full, folds, num_rows, target_one_hot, scaler, timesteps, subset, baseline_energy):
    update_from = energy_full.filter(subset)
    update_to = energy.copy()
    update_to.update(update_from)
    test_fold = make_test_fold_fast(update_to, folds[-1][-1], num_rows, target_one_hot, scaler, timesteps)

    retrofit_energy = np.sum(model.predict(test_fold))

    return 100 * (retrofit_energy / baseline_energy - 1)

def stepwise_selection(model, energy, retrofit_df, folds, test_folds, scaler, timesteps, savings_path):
    
    df = energy.query(folds[-1][-1])
    num_rows = df['apn'].value_counts().min() // timesteps * timesteps
    df = one_hot_encode(df.groupby('apn').head(num_rows), 'apn', 'target')
    target_one_hot = df.filter(regex='^target', axis=1)
    series_y = df['kwh_actual'].copy().reset_index(drop=True).apply(lambda x: np.where(x < 1, 1, x))
    baseline_energy = np.sum(model.predict(test_folds[-1][0]))

    all_columns = np.array(retrofit_df.filter(regex='^kwh_sim', axis=1).columns)
    
    building_pool = list(all_columns)
    building_set = []
    savings_history = []

    total_savings = 0

    while True:
        best = 0
        for building in building_pool:
            building_set.append(building)
            saving = calc_savings(model, energy, retrofit_df, folds, num_rows, target_one_hot, scaler, timesteps, building_set, baseline_energy)
            saving_diff = saving - total_savings
            if saving_diff < best:
                best = saving_diff
                best_building = building
            building_set.pop()
        if best == 0:
            break
        total_savings += best
        savings_history.append(best)
        building_set.append(best_building)
        building_pool.remove(best_building)

        if not building_pool:
            break

    all_savings = pd.DataFrame({'building': building_set, 'savings': savings_history})
    all_savings.to_csv(savings_path, index=False)
    
def stepwise_selection_no_context(retrofits_df, retrofit_column, savings_path):
    df = retrofits_df.copy()
    df = df.filter(regex='(^kwh_|apn)').groupby('apn').agg('sum').reset_index()
    df.sort_values(retrofit_column, inplace=True)
    
    baseline_energy = sum(df['kwh_baseline'])
    
    savings = []
    for apn in df['apn'].unique():
        other = sum(df[df['apn'] != apn]['kwh_baseline'])
        retrofit_energy = other + df[df['apn'] == apn][retrofit_column]
        saving = 100 * (retrofit_energy / baseline_energy - 1)
        savings.append(float(saving))

    all_savings = pd.DataFrame({'building': 'kwh_sim_' + df['apn'].unique(), 'savings': savings})
    all_savings.sort_values('savings', inplace=True)
    all_savings.query('savings < 0', inplace=True)
    all_savings.to_csv(savings_path, index=False)
    
def plot_stepwise_history(stepwise_history, elbow, retrofit):
    plt.title("% Change in Urban Energy Use - " + retrofit + " Retrofit")
    plt.xlabel("# of Retrofitted Buildings")
    plt.ylabel("% Change")
    plt.plot(range(1, 1 + len(stepwise_history)), np.array(stepwise_history['savings']), '-o')
#     plt.xlim([0, 25])
#     plt.ylim([-3, 0])
    plt.show()
    plt.title("Cumulative % Change in Urban Energy Use - " + retrofit + " Retrofit")
    plt.xlabel("# of Retrofitted Buildings")
    plt.ylabel("% Change")
    plt.axvline(elbow, color='0', linestyle='--')
    plt.plot(range(1, 1 + len(stepwise_history)), np.cumsum(stepwise_history['savings']), '-o')
#     plt.xlim([0, 25])
#     plt.ylim([-14, 0])
    plt.show()