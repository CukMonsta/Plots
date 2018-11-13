# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

################################################ Evaluate the Fit of the Model
# Generate a dataset
samples = 50

function = lambda X: np.cos(1.5 * np.pi * X)

X = np.sort(np.random.rand(samples))
y = function(X) + np.random.randn(samples) * 0.3

# Generate a function to set the degree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def evaluateFit(degrees, X, y):
    polynomial_features = PolynomialFeatures(degree = degrees, include_bias = False)
    
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)
    
    scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring = "neg_mean_squared_error", cv = 10)
    
    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, function(X_test), label = "function")
    plt.scatter(X, y, label = "Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees, -scores.mean(), scores.std()))
    plt.show()

# Test the function    
evaluateFit(2, X, y)

