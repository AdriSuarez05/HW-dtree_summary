# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 16:27:04 2022

@author: Adriana Suarez
"""
"""
email: adriana.suarezb@upb.edu.co
ID: 502197

"""
import statsmodels.api as sm 
import numpy as np
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
import matplotlib.image as plting
import matplotlib.pyplot as plt

carseats = sm.datasets.get_rdataset("Carseats", "ISLR")
datos = carseats.data
print(carseats.__doc__) 

datos['ventas_altas'] = np.where(datos.Sales > 8, 0, 1) 
datos = datos.drop(columns = 'Sales')
print(datos.dtypes)

#Reemplazamos los valores String por valores numericos
# Creamos el diccionario para la variable sheveloc
d = {'Bad': 0, 'Medium': 1, 'Good': 2}
# Mapeamos la variable sheveloc y reemplazamos el valor del directorio
datos['ShelveLoc'] = datos['ShelveLoc'].map(d)

#Reemplazamos los valores String por valores numericos
# Creamos el diccionario para la variable Urban
d = {'Yes': 1, 'No': 0}
# Mapeamos la variable Urban y reemplazamos el valor del directorio
datos['Urban'] = datos['Urban'].map(d)

#Reemplazamos los valores String por valores numericos
# Creamos el diccionario para la variable US
d = {'Yes': 1, 'No': 0}
# Mapeamos la variable US y reemplazamos el valor del directorio
datos['US'] = datos['US'].map(d)


features = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'ShelveLoc', 'Age', 'Education', 'Urban', 'US' ]
X = datos[features]
y = datos['ventas_altas']



# #Se crea el arbol de decision
# dtree  = DecisionTreeClassifier()
# dtree = dtree.fit(X, y)

# print(dtree.predict([[132, 110, 0, 108, 124, 1, 76, 10, 0, 0]]))

# print("[1] means 'Bajo'")
# print("[0] means 'Alto'")

# # Para mostrar la grafica de arbol

# datos = tree.export_graphviz(dtree, out_file = None, feature_names=features)
# graph = pydotplus.graph_from_dot_data(datos)
# graph.write_png('mydtreea.png')

# img = plting.imread('mydtreea.png')
# imgplot = plt.imshow(img)
# plt.show()

# ----------------------------------- TRAIN/TEST-------------------------------
# Divida sus datos en train / test
# X_2 = datos[features]
# y_2 = datos['ventas_altas'] / X_2

# Se hace la division de los datos en 80% para train y 20% para test
# Train 80%
# 

# Modelo de regresion polinomial

# mymodel = np.poly1d(np.polyfit(train_x, train_y, 4))

# # R2 cuadrado Verificamos el valor de r2 score
# r2 = r2_score(train_y, mymodel(train_x))
# print(r2)

# # Linea para graficar
# myline = np.linspace(132, 110, 0, 108, 124, 1, 76, 10, 0, 0)

# plt.scatter(train_x, train_y)
# plt.plot(myline, mymodel(myline))
# plt.show()

# # Evaluar el modelo usando los datos de prueba (test-20%)
# r2_test = r2_score(test_y, mymodel(test_x))
# print(r2_test)


#Predicción 
#print(mymodel(132, 110, 0, 108, 124, 1, 76, 10, 0, 0))












