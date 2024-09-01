import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Ian\\Desktop\\Estudos\\Estudos Kaggle\\Intro ML\\carros.csv')

Q1 = df['selling_price'].quantile(0.25)
Q3 = df['selling_price'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Remover os outliers
df = df[(df['selling_price'] >= limite_inferior) & (df['selling_price'] <= limite_superior)]

Q1 = df['km_driven'].quantile(0.25)
Q3 = df['km_driven'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Remover os outliers
df = df[(df['km_driven'] >= limite_inferior) & (df['km_driven'] <= limite_superior)]

# sns.boxplot(y=df['km_driven'])
# plt.show()
# sns.boxplot(y=df['selling_price'])
# plt.show()

df['log_selling_price'] = np.log1p(df['selling_price'])
df['log_km_driven'] = np.log1p(df['km_driven'])

df = df[(df['fuel'] == 'Diesel') | (df['fuel'] == 'Petrol')]
df = df[df['seller_type'] != 'Trustmark Dealer']
df = df[(df['owner'] != 'Fourth & Above Owner') & (df['owner'] != 'Test Drive Car')]

label_encoder = LabelEncoder()

df['novo_fuel'] = label_encoder.fit_transform(df['fuel'])
df['novo_seller_type'] = label_encoder.fit_transform(df['seller_type'])
df['novo_owner'] = label_encoder.fit_transform(df['owner'])
df['novo_transmission'] = label_encoder.fit_transform(df['transmission'])
df['age'] = 2024 - df['year']

x = df[['age', 'log_km_driven', 'novo_fuel', 'novo_transmission', 'novo_owner', 'novo_seller_type']]
y = df['log_selling_price']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=0)

rf = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=0)
rf.fit(x_treino, y_treino)

y_predict = rf.predict(x_teste)

acuracia = mean_absolute_error(np.expm1(y_predict), np.expm1(y_teste))

print(f'erro médio absoluto: {acuracia} \nImportância das features:')

importances = rf.feature_importances_
feature_names = x.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print(feature_importance_df)