# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-1 : Import the required library and read the dataframe.

Step-2 : Write a function computeCost to generate the cost function.

Step-3 : Perform iterations og gradient steps with learning rate.

Step-4 : Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: CHANDRAPRIYADHARSHINI C
RegisterNumber: 212223240019
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]

  theta=np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)

    errors=(predictions-y).reshape(-1,1)

    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)

  return theta
data=pd.read_csv("/content/50_Startups.csv")
data.head()

x=(data.iloc[:,:-2].values)
x1=x.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x)
print(x1_scaled)

theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted valeue: {pre}")
```

## Output:
![image](https://github.com/Bosevennila/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870486/8524eb7b-13f6-4a1e-b9db-9b002eb0ec7a)

![image](https://github.com/Bosevennila/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870486/527a54b8-6848-44e8-a5c8-757abc6306a7)

![image](https://github.com/Bosevennila/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870486/13a6007d-0c99-4148-9548-01f8faba1ac1)

predicted value
![image](https://github.com/Bosevennila/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870486/27eb69ab-ad61-4d78-aa9c-0d8b4c74947c)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
