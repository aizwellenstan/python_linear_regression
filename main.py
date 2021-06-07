import numpy as np
import matplotlib.pylot as plt
import padnas as pd
import seaborn as sns

stocks = pd.read_csv('./.csv')
x = stocks.iloc[:, :-1].values
y = stocks.iloc[:, 4].values

sns.heatmap(stocks.corr())

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarry()

X = X[:, 1:]

form sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.liner_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictiong the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)

# CalcuLating the Coefficients
print(regressor.coef_)

# CalcuLating the Intercept
print(regressor.intercept_)

from skklear.matrics import r2_score
r2_score(y_test, y_pred)
