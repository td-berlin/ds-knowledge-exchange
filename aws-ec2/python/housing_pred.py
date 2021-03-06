import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
import os
import argparse

parser = argparse.ArgumentParser(description="Example for DS Knowledge Sharing using AWS and EC2")
parser.add_argument("-a", "--algorithm", required = True, choices=["linear","ridge","polyl2","svr", "tree"], default="linear",
                    help="select an algorithm to benchmark [linear|svm|tree|ridge|polyl2]")
parser.add_argument("-v", "--verbose", default=0, help="Should I be verbose?")
parser.add_argument("-p", "--input_path", required=True, help="Input path and file")
parser.add_argument("-o", "--output_path", required=True, help="Output path and file")
args = parser.parse_args()

alg = args.algorithm
input_path = args.input_path
output_path = args.output_path
verbose = bool(int(args.verbose))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
#Lets load the dataset and sample some
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(input_path, header=None, delimiter=r"\s+", names=column_names)

column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = data.loc[:,column_sels]
y = data['MEDV']

min_max_scaler = preprocessing.MinMaxScaler()
if(verbose): print('minmax::Scaling')
x_scaled = min_max_scaler.fit_transform(x)

if(verbose): print('kfold::preparding folds')
kf = KFold(n_splits=10)

scores_map = {}
if(alg == 'linear'):
    l_regression = linear_model.LinearRegression()
    if(verbose): print('linear::Cross validating')
    scores = cross_val_score(l_regression, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    
    scores_map['LinearRegression'] = scores

if(alg == 'ridge'):
    l_ridge = linear_model.Ridge()
    if(verbose): print('ridge::Cross validating')
    scores = cross_val_score(l_ridge, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')

    scores_map['Ridge'] = scores
    
if(alg == 'polyl2'):
    # Lets try polinomial regression with L2 with degree for the best fit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    #for degree in range(2, 6):
    #    model = make_pipeline(PolynomialFeatures(degree=degree), linear_model.Ridge())
    #    scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    #    print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge())
    if(verbose): print('polyl2::Cross validating')
    scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    scores_map['PolyRidge'] = scores

if(alg == 'svr'):
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    
    if(verbose): print('SVR::Initiating SVR')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    #grid_sv = GridSearchCV(svr_rbf, cv=kf, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, scoring='neg_mean_squared_error')
    #grid_sv.fit(x_scaled, y)
    #print("Best classifier :", grid_sv.best_estimator_)
    if(verbose): print('SVR::Cross validating')
    scores = cross_val_score(svr_rbf, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    scores_map['SVR'] = scores


if(alg == 'tree'):
    from sklearn.tree import DecisionTreeRegressor
    
    desc_tr = DecisionTreeRegressor(max_depth=5)
    #grid_sv = GridSearchCV(desc_tr, cv=kf, param_grid={"max_depth" : [1, 2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
    #grid_sv.fit(x_scaled, y)
    #print("Best classifier :", grid_sv.best_estimator_)
    scores = cross_val_score(desc_tr, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    scores_map['DecisionTreeRegressor'] = scores

print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

f = open(output_path, "a")
f.write("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
f.close()

