import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sktime.forecasting.compose import make_reduction
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, mean_absolute_scaled_error, MeanSquaredError
import pickle

#Reading Datasets
Dataset_WF_site1 = pd.read_excel(r'C:\Users\udish\OneDrive\Documents\Udisha\Northeastern\Spring 2024\ML_IoT\ML Model Deployment\Wind farm site 1 (Nominal capacity-99MW).xlsx'
                                ).drop(index=0)
Dataset_WF_site2 = pd.read_excel(r'C:\Users\udish\OneDrive\Documents\Udisha\Northeastern\Spring 2024\ML_IoT\ML Model Deployment\Wind farm site 2 (Nominal capacity-200MW).xlsx'
                                ).drop(index=0)
Dataset_WF_site2.rename(columns={'Wind speed - at the height of wheel hub  (m/s)': 'Wind speed - at the height of wheel hub (m/s)'}, inplace=True)
Dataset_WF_site3 = pd.read_excel(r'C:\Users\udish\OneDrive\Documents\Udisha\Northeastern\Spring 2024\ML_IoT\ML Model Deployment\Wind farm site 3 (Nominal capacity-99MW).xlsx'
                                ).drop(index=0)


def interpolate_zeros(df):
    df_zero_rows = (df == 0).any(axis=1)
    df.loc[df_zero_rows] = df.loc[df_zero_rows].replace(0, np.nan).interpolate('ffill')
    return df

def calc_err_metrics(pred, train=None, test=None):
    print("MAPE: ", mean_absolute_percentage_error(test, pred))
    print("MAE: ", mean_absolute_error(test, pred))
    print("MASE: ", mean_absolute_scaled_error(test, pred, y_train=train))
    print("MSE: ", mean_squared_error(test.squeeze(), pred.squeeze()))
    # rmse = MeanSquaredError(square_root=True)
    # print("RMSE: ", rmse(test.squeeze(), pred.squeeze()))
    
def plot_data(test, num_hours):
    plt.plot(test['Power (MW)'][:4*num_hours], label='Actual')
    plt.plot(test['Pred'][:4*num_hours], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Power (MW)')
    plt.legend()
    plt.show()

def checkVIF(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    vif = pd.DataFrame()
    vif['Features'] = numeric_cols
    vif['VIF'] = [variance_inflation_factor(X[numeric_cols].values, i) for i in range(len(numeric_cols))]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)

def create_features(df):
    df = df.copy()
    df['power_lag1'] = df['Power (MW)'].shift(1).fillna(method='bfill')
    df['power_lag2'] = df['Power (MW)'].shift(2).fillna(method='bfill')
    df['power_lag3'] = df['Power (MW)'].shift(3).fillna(method='bfill')
    return df

def classify_wind_speed(df, wind_speed_column):
    bins = [0, 5, 10, 15, 20, 25, np.inf]
    labels = ['1', '2', '3', '4', '5', '6']
    df['WindSpeedTier'] = pd.cut(df[wind_speed_column], bins=bins, labels=labels)
    df['WindSpeedTier'] = df['WindSpeedTier'].astype('category')
    return df

def predict(forecaster, X_test, y_test, num_hours):
    y_pred = []
    updateEvery = 2
    for i in range(1, (4*num_hours)+1):
        y_pred_i = forecaster.predict(fh=1, X=X_test.iloc[[i]])
        y_pred.append(y_pred_i.iloc[0])
        if i % updateEvery == 0:
            forecaster.update(y_test.iloc[:i+1], X=X_test.iloc[:i+1])

    y_pred = pd.Series(y_pred, index=y_test.iloc[1:(4*num_hours)+1].index)
    return y_pred

def preprocess_data(df):
    Features = ['Wind speed - at the height of wheel hub (m/s)']
    Target = ['Power (MW)']

    df2 = df[Features + Target].copy()
    df2 = create_features(df2)
    df2 = classify_wind_speed(df2, 'Wind speed - at the height of wheel hub (m/s)')
    df2 = df2.set_index(pd.to_datetime(df['Time(year-month-day h:m:s)']))
    Features = df2.columns.difference(['Power (MW)'])

    df2 = interpolate_zeros(df2)
    df2.fillna(method='ffill', inplace=True)
    
    train_size = int(len(df2) * 0.75)
    train, test = df2.iloc[:train_size], df2.iloc[train_size:]
    X_train = train[Features]
    y_train = train[Target]
    X_test = test[Features]
    y_test = test[Target]

    return X_train, y_train, X_test, y_test

    
def create_model(df, win_size, strategy):
    X_train, y_train, _, _= preprocess_data(df)
    lm = LinearRegression()
    forecaster = make_reduction(lm, window_length=4*win_size, strategy=strategy, windows_identical=False)
    forecaster.fit(y_train, X=X_train, fh=1)
    return forecaster

def predict_with_model(forecaster, df, num_hours):
    _, _, X_test, y_test = preprocess_data(df)
    y_pred = predict(forecaster, X_test, y_test, num_hours)
    test = y_test.copy()
    test['Pred'] = y_pred
    return test

def calculate_metrics(df,test, num_hours):
    X_train, y_train, _, _= preprocess_data(df)
    return calc_err_metrics(test=test['Power (MW)'][1:4*num_hours].to_frame(), pred=test['Pred'][1:4*num_hours].to_frame(), train=y_train['Power (MW)'].to_frame())
    
    

model_Farm_1 = create_model(Dataset_WF_site1, 6, 'recursive')
model_Farm_2 = create_model(Dataset_WF_site2, 6, 'recursive')
model_Farm_3 = create_model(Dataset_WF_site3, 6, 'recursive')  
# Make pickle file of our model

pickle.dump(model_Farm_1, open("model_F1.pkl", "wb"))
pickle.dump(model_Farm_2, open("model_F2.pkl", "wb"))
pickle.dump(model_Farm_3, open("model_F3.pkl", "wb"))
