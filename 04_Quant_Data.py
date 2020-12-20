import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
import urllib.request
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% Download macro data -------------------------------------------------------------------------------------

GDP_url = "https://sdw.ecb.europa.eu/quickviewexport.do;jsessionid=44DBD7559D119BEF9DA13CCEB70C23E1?SERIES_KEY=MNA.Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N&type=csv"
Eurlibor_url = "https://sdw.ecb.europa.eu/quickviewexport.do;jsessionid=0014F48DEAEA2500294F6E3962E49775?SERIES_KEY=143.FM.M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA&type=csv"
Consumption_url = "https://sdw.ecb.europa.eu/quickviewexport.do;jsessionid=1573D4C068E9F1CA1116D3998B81380E?SERIES_KEY=MNA.Q.Y.I8.W0.S1M.S1.D.P31._Z._Z._T.EUR.V.N&type=csv"
GovSpend_url = "https://sdw.ecb.europa.eu/quickviewexport.do;jsessionid=3F35BD60FBC44B0E9380EFA27A0622CF?SERIES_KEY=MNA.Q.Y.I8.W0.S13.S1.D.P3._Z._Z._T.EUR.V.N&type=csv"
Export_url = "https://sdw.ecb.europa.eu/quickviewexport.do;jsessionid=5A4DE9E92D17B294D6A5315EE8355389?SERIES_KEY=MNA.Q.Y.I8.W1.S1.S1.D.P6._Z._Z._Z.EUR.V.N&type=csv"
Import_url = "https://sdw.ecb.europa.eu/quickviewexport.do;jsessionid=E521FE1B420F2DADB8A4B3B88B00B6A9?SERIES_KEY=MNA.Q.Y.I8.W1.S1.S1.C.P7._Z._Z._Z.EUR.V.N&type=csv"

# GDP typically consisty of Y = C + I + G + (X-M)
urllib.request.urlretrieve(GDP_url, "data/GDP_data.csv")
urllib.request.urlretrieve(Eurlibor_url, "data/Eurlibor_data.csv")
urllib.request.urlretrieve(Consumption_url, "data/Consumption_data.csv")
urllib.request.urlretrieve(GovSpend_url, "data/GovSpend_data.csv")
urllib.request.urlretrieve(Export_url, "data/Export_data.csv")
urllib.request.urlretrieve(Import_url, "data/Import_data.csv")

GDP_data = pd.read_csv('data/GDP_data.csv', skiprows=range(1, 5))
Eurlibor_data = pd.read_csv('data/Eurlibor_data.csv', skiprows=range(1, 5))
Consumption_data = pd.read_csv('data/Consumption_data.csv', skiprows=range(1, 5))
GovSpend_data = pd.read_csv('data/GovSpend_data.csv', skiprows=range(1, 5))
Export_data = pd.read_csv('data/Export_data.csv', skiprows=range(1, 5))
Import_data = pd.read_csv('data/Import_data.csv', skiprows=range(1, 5))

# Convert monthly Eurlibor to quarterly data
# df.rename(index=lambda s: s[4:] + '!!')
# Eurlibor_data #TODO: convert + add Quarterly EURLIBOR


# %% Data Cleaning

GDP_data.shape
Eurlibor_data.shape
Consumption_data.shape
GovSpend_data.shape
Export_data.shape
Import_data.shape

# Convert GDP to GDP_growth for prediction
#GDP_gr_data = GDP_data.multiply(10 ^ 6).pct_change(-1).add(1).pow(4).add(-1)

# Construct final DataFrame
data = pd.concat([GDP_data, Consumption_data, GovSpend_data, Export_data, Import_data], axis=1)
data.columns = ["GDP", "Consumption", "GovSpend", "Exports", "Imports"]

# remove NA caused by growth rate calculation
data = data.dropna()
data.isna().sum()

# %% Model

# split data into X / Y
#Y = data['GDP_gr']
Y = data['GDP']
X = data.loc[:, data.columns != 'GDP']
#X = StandardScaler().fit_transform(X)  # Standardize

# Split data into train / test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2020)

alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
elastic_cv = ElasticNetCV(alphas=alphas, cv=5)

model = elastic_cv.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)

print("R2:{0:.4f}, MSE:{1:.4f}, RMSE:{2:.4f}".format(score, mse, np.sqrt(mse)))

# %% Plot results

x_ax = range(len(X_test))
plt.scatter(x_ax, y_test, s=5, color='blue', label='original')
plt.plot(x_ax, y_pred, lw=0.8, color='red', label='predicted')
plt.legend()
plt.show()

GDP_data.plot()
Eurlibor_data.plot()
plt.show()
