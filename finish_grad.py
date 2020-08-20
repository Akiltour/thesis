import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance
import random 
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics

dataset = pd.read_csv('finish_noise.csv',sep=';')

#print(dataset_test.shape)
#print(dataset_test.describe())
#print(dataset_test.isnull().any())

print(dataset.shape)
print(dataset.describe())
#print(dataset.isnull().any())

X = dataset[['cg09809672', 'cg22736354', 'cg02228185', 'cg01820374', 'cg06493994', 'cg19761273']].values
y = dataset['Age'].values

params = {'n_estimators': 200,
          'max_depth': 3,
          'learning_rate': 0.1,
          'criterion': 'mse'}

#noise
total = 0
noised = 0
for i in range(len(X)):
	for j in range(len(X[i])):
		num = random.randint(1, 10)
		if num <= 5:
			noised += 1
			#mu, sigma = 0, 0.1 
			#noise = np.random.normal(mu, sigma, [1])
			#print(X[i][j])
			#X[i][j] = X[i][j] + noise
			X[i][j] = 0
			#print("noise : ", noise)
			#print(X[i][j]
		total += 1
		#print(X[i][j])

#print("noised :  ",noised)
#print("total : ", total)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
regressor = ensemble.GradientBoostingRegressor(**params)
regressor.fit(X_train,y_train)



y_pred = regressor.predict(X_test)
df = pd.DataFrame({'True age': y_test, 'Predicted age': y_pred})
df1 = df.head(1400)
print(df1)


df1.plot(x='True age', y='Predicted age', style='o') 
x = np.linspace(10,100,90)
y = x
plt.plot(x, y, '-r', label='y=x')
plt.title('Gradiant boosting')  
plt.xlabel('True age')  
plt.ylabel('Predicted age')  
plt.show()


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
"""
#mse = mean_squared_error(y_test, regressor.predict(X_test))
#print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
#print(regressor.score(X,y))

#y_pred = regressor.predict(X_test)
plt.figure(figsize=(12,6))
plt.title("test")
print(len(X))
print(len(y))
plt.scatter(X,y)
plt.plot(X,regressor.predict(X), color='black')
plt.show()
"""


"""
regressor = LinearRegression()  
regressor.fit(X, y)

#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
#print(coeff_df)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'True age': y_test, 'Predicted age': y_pred})
df1 = df.head(104)
print(df1)

#df1.plot(kind='bar',figsize=(10,8))
#plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()

df1.plot(x='True age', y='Predicted age', style='o')  
x = np.linspace(10,70,60)
y = x
plt.plot(x, y, '-r', label='y=x')
plt.title('True age vs Predicted age')  
plt.xlabel('True age')  
plt.ylabel('Predicted age')  
plt.show()


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

"""





