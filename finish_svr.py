import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
import random
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn import metrics

dataset = pd.read_csv('finish_noise.csv',sep=';')

print(dataset.shape)
print(dataset.describe())
print(dataset.isnull().any())

X = dataset[['cg09809672', 'cg22736354', 'cg02228185', 'cg01820374', 'cg06493994', 'cg19761273']].values
y = dataset['Age'].values

#noise
"""
total = 0
noised = 0
for i in range(len(X)):
	for j in range(len(X[i])):
		num = random.randint(1, 10)
		if num <= 3:
			noised += 1
			mu, sigma = 0, 0.1 
			noise = np.random.normal(mu, sigma, [1])
			#print(X[i][j])
			#X[i][j] = X[i][j] + noise
			X[i][j] = 0
			#print("noise : ", noise)
			#print(X[i][j])
		total += 1
		#print(X[i][j])

#print("noised :  ",noised)
#print("total : ", total)
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)
df = pd.DataFrame({'True age': y_test, 'Predicted age': y_pred})
df1 = df.head(1400)
print(df1)


df1.plot(x='True age', y='Predicted age', style='o')  
x = np.linspace(10,100,90)
y = x
plt.plot(x, y, '-r', label='y=x')
plt.title('SVR regression')  
plt.xlabel('True age')  
plt.ylabel('Predicted age')  
plt.show()

print('R²:', r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))







