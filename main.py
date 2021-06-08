import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

try:
  stocks = pd.read_csv('./screener.csv')
  df = pd.DataFrame(stocks)
  for column in df:
    if(column!="symbol"):
      df = df[df[column].notnull()]
      
  df = df[['3month_performance', 'avgvol_10']]

  df['avgvol_10'] = df['avgvol_10'].replace(r'[KM]+$', '', regex=True)
  # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  #   print(df)
  df.to_csv('./symbolLst.csv')
  # df['avgvol_10'] = (df['avgvol_10'].replace(r'[KM]+$', '', regex=True).astype(float) * \
  #   df['avgvol_10'].str.extract(r'[\d\.]+([KM]+)', expand=False)
  #     .fillna(1)
  #     .replace(['K','M'], [10**3, 10**6]).astype(int))
  # print(df)
  corr = df.corr()
  plt.figure()
  sns.heatmap(corr, square=True, annot=True)
  plt.savefig('./seaborn_heatmap_list.png')
  plt.close('all')

  # print(corr)
except Exception as e:
  print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
  print(e)


# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder = LabelEncoder()
# X[:, 3] = labelencoder.fit_transform(X[:, 3])

# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarry()

# X = X[:, 1:]

# form sklearn.model_selection import train_test_split
# X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# from sklearn.liner_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# # Predictiong the Test set results
# y_pred = regressor.predict(X_test)
# print(y_pred)

# # CalcuLating the Coefficients
# print(regressor.coef_)

# # CalcuLating the Intercept
# print(regressor.intercept_)

# from skklear.matrics import r2_score
# r2_score(y_test, y_pred)
