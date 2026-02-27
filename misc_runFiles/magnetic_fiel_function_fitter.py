import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class Magnetic_function_fitter():

    def __init__(self, file):

        self.magnetic_data = pd.read_csv(file)
        self.x = self.magnetic_data['x']
        self.y = self.magnetic_data['y']
        self.z = self.magnetic_data['z']
        #print(self.magnetic_data)


    def fitter(self, degree, B):

        B = self.magnetic_data[B]

        cart_coord = np.column_stack((self.x, self.y, self.z))
        #print(cart_coord)
        poly = PolynomialFeatures(degree=degree)

        X_poly = poly.fit_transform(cart_coord)
        #print(X_poly)
        model = LinearRegression().fit(X_poly, B)
        #print(model)
        fit_coeffs  = model.coef_
        #print(X_poly[0])
        fit_coeffs[0] = model.intercept_
        #print(fit_coeffs)
        #for xs in X_poly:
            #print(np.dot(xs,  fit_coeffs))

        #print(B)

        return cart_coord, X_poly, fit_coeffs

    def fit_tester(self, fit_profile, B):

        fig, ax = plt.subplots()
        num_points = np.linspace(0, len(fit_profile)-1, len(fit_profile))
        ax.scatter(num_points, self.magnetic_data[B], color = 'k', marker = 'x')
        ax.set_xlim(950, 1000)
        ax.scatter(num_points, fit_profile, color = 'r', marker = 'D')



