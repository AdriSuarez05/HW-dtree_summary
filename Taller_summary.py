# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 08:24:58 2022

@author: Adriana Suarez
"""
"""
email: adriana.suarezb@upb.edu.co
ID: 502197

"""
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()


car_data = pd.read_csv('cars2.csv')


# Estandarizacion para la tabla car2

x_car = car_data[['Weight']]
y_car = car_data['CO2'] 

scaledX = scale.fit_transform(x_car)
print(" Estandarización tabla de carros ")
print(scaledX)
print("    ")


#Diagrama de dispersion dataframe cars2
plt.scatter(x_car, y_car)
plt.title('Car2')
plt.show()

#Para la tabla Cars2 ------------------------------------------------------------------------
# Se hace la division de los datos en 80% para train y 20% para test para car_data
#Train 80%
train_x = x_car[:28]
train_y = y_car[:28]

#Test 20%
test_x = x_car[28:]
test_y = y_car[28:]

# Mostrar los datos segmentados en el entrenamiento con diagramas de dispersion
plt.scatter(train_x, train_y)
plt.show()

# Mostrar los datos segmentados en la prueba con diagramas de dispersion
plt.scatter(test_x, test_y)
plt.show()

# Regresion Multiple, debido a la forma de las graficas se decide hacer regresion multiple
reg_mod = linear_model.LinearRegression()
reg_mod.fit(train_x, train_y)

# Prediccion
predic_co2 = reg_mod.predict([[3300]])
print(" Predicción de la tabla cars:")
print(predic_co2)

# Se ajusta modelo a los datos escalados
# Predecir el comportamiento de la variable dependiente – Usando valores estandarizados.
reg_mod.fit(scaledX, y_car)

scaled = scale.transform([[2300]])

predictedCO2 = reg_mod.predict([scaled[0]])
print(" Predicción con los valores estandarizados:")
print(predictedCO2)

# Calcular el r de relación
r2 = reg_mod.score(scaledX, y_car)
print("R de relación para cars: ")
print(r2)

#%%

# Student----------------------------------------------------------------------

import pandas as pd
import numpy
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

student_data = pd.read_csv('student_data.csv')

# Estandarizacion para la tabla student
x_student = student_data[['age', 'G2', 'freetime' ]]
y_student = student_data[['Medu', 'G3', 'studytime']]

scaledX = scale.fit_transform(x_student)
print(" Estandarización tabla de estudiantes: ")
print(scaledX)

# Diagrama de dispersion dataframe student
plt.scatter(x_student, y_student)
plt.title('Student')
plt.show() 

# Se hace la division de los datos en 80% para train y 20% para test para car_data
#Train 80%
train_x2 = x_student[:315]
train_y2 = y_student[:315]

#Test 20%
test_x2 = x_student[315:]
test_y2 = y_student[315:]

# Mostrar los datos segmentados en el entrenamiento con diagramas de dispersion
plt.scatter(train_x2, train_y2)
plt.show()

# Mostrar los datos segmentados en la prueba con diagramas de dispersion
plt.scatter(test_x2, test_y2)
plt.show()

# Regresion Multiple, debido a la forma de las graficas se decide hacer regresion multiple
reg_mod2 = linear_model.LinearRegression()
reg_mod2.fit(train_x2, train_y2)

# Prediccion
predic_co2_2 = reg_mod2.predict([[18, 5, 4]])
print(" Predicción de la tabla student:")
print(predic_co2_2)

# Se ajusta modelo a los datos escalados
# Predecir el comportamiento de la variable dependiente – Usando valores estandarizados.
reg_mod2.fit(scaledX, y_student)

scaled2 = scale.transform([[18, 5, 4]])

predictedCO2_2 = reg_mod2.predict([scaled2[0]])
print(" Predicción con los valores estandarizados:")
print(predictedCO2_2)

# Calcular el r de relación
r2_2 = reg_mod2.score(scaledX, y_student)
print("R de relación para student: ")
print(r2_2)

#%%

import pandas as pd
import numpy
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()


netflix_data = pd.read_csv('netflix_titles.csv')
print(netflix_data.dtypes)

# Estandarizacion para la tabla netflix
x = netflix_data[['release_year']][0 : 1001]

netflix_data["duracion"] = pd.to_numeric(netflix_data['duration'].replace('([^0-9]*)','', regex=True), errors='coerce')

y = netflix_data['duracion'][0 : 1001]


scaledX = scale.fit_transform(x)
print(" Estandarización tabla de netflix: ")
print(scaledX)

# Diagrama de disersion dataframe netflix
plt.scatter(x, y)
plt.title('Netflix')
plt.show() 

# Se hace la division de los datos en 80% para train y 20% 
#Train 80%
train_x3 = x[:800]
train_y3 = y[:800]

#Test 20%
test_x3 = x[:200]
test_y3 = y[:200]

# Mostrar los datos segmentados en el entrenamiento con diagramas de dispersion
plt.scatter(train_x3, train_y3)
plt.show()

# Mostrar los datos segmentados en la prueba con diagramas de dispersion
plt.scatter(test_x3, test_y3)
plt.show()

# # Regresion Multiple, debido a la forma de las graficas se decide hacer regresion multiple
reg_mod3 = linear_model.LinearRegression()
reg_mod3.fit(train_x3, train_y3)

# Calcular el r de relación
r2 = reg_mod3.score(scaledX, y)
print("R de relación para netflix: ")
print(r2)

# Se ajusta modelo a los datos escalados
# Predecir el comportamiento de la variable dependiente – Usando valores estandarizados.
reg_mod3.fit(scaledX, y)

scaled3 = scale.transform([[1993]])

predictedCO2_3 = reg_mod3.predict([scaled3[0]])
print(" Predicción con los valores estandarizados:")
print(predictedCO2_3)




