import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Import the data on percentages of EVs
data = pd.read_csv('other data\percentage_EVs_EHV.csv',sep=';',index_col=False)[['Periode', 'Waarde']].iloc[:5]
# Convert the years to datetime format
data['Date'] = data['Periode'].apply(lambda x: pd.to_datetime(str(x) + '-1-1'))
# Convert the datetimes to days for regression
data['Date_days'] = (data['Date'] - pd.Timestamp("2019-01-01")).dt.days
# Change the float notation
data['Waarde'] = data['Waarde'].str.replace(',','.').astype(float)

# Split the data, since the dataset is small there is only one row in the test set
X = data[['Date_days']]
y = data['Waarde']
X_train = X.iloc[:-1]
X_test = X.iloc[[-1]]
y_train = y.iloc[:-1]
y_test = y.iloc[[-1]]

# Convert y to log for exponential regression
y_log_train = np.log(y_train)
# Create, train and predict the test set with the exponential model
exp_model = LinearRegression()
exp_model.fit(X_train, y_log_train)
y_exp_pred = np.exp(exp_model.predict(X_test))
# Evaluate performance on the test set
print('Absolute error: ', abs(y_exp_pred - y_test.iloc[0]))

# Fit the model to all of the data for better future prediction
exp_model.fit(X,np.log(y))

# Initialize future years to predict and convert them to the right format
future_years = ['2026-1-1', '2027-1-1', '2028-1-1']
future_years_dt = []
exp_future_preds = []

for i in future_years:
    year_datetime = pd.to_datetime(i)
    future_years_dt.append(year_datetime)
    converted_year = (year_datetime - pd.Timestamp("2019-01-01")).days
    # Predict the values for the future years
    exp_future_preds.append(np.exp(exp_model.predict([[converted_year]]))[0])
    
# Create a list of all dates to display in the visualization
all_dates = list(data['Date']) + future_years_dt
# Create the curve to display the exponential regression (ER)
all_dates_numeric = [(d - pd.Timestamp("2019-01-01")).days for d in all_dates]
exp_line = np.exp(exp_model.predict(pd.DataFrame({'Date_days': all_dates_numeric})))

# Plot the data, prediction of the test set, predictions for the future, and the ER curve
plt.figure()
plt.scatter(data['Date'],y,c='black', label='Real data')
plt.scatter(data['Date'].iloc[-1], y_exp_pred, c='red', label='Prediction of test set')
plt.scatter(future_years_dt, exp_future_preds, c='blue', label='Exponential predictions')
plt.plot(all_dates, exp_line, label='Fitted ER line')
plt.title('Prediction of Percentage of EVs and Plug-in Hybrid Cars in Eindhoven')
plt.xlabel('Date')
plt.ylabel('Percentage of EVs and plug-in hybrid cars')
plt.legend()
plt.show()

# Print the future predictions' values for easy access
result_df = pd.DataFrame({'Date':future_years, 'Exp_prediction':exp_future_preds})
print(result_df)