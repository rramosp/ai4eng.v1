import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd
import warnings

from bokeh.plotting import *
#from bokeh.charts import *
from bokeh.models import *
import bokeh

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

output_notebook(resources=bokeh.resources.INLINE)
warnings.filterwarnings('ignore')


def latlng_to_meters(lat, lng):
    origin_shift = 2 * np.pi * 6378137 / 2.0
    mx = lng * origin_shift / 180.0
    my = np.log(np.tan((90 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    my = my * origin_shift / 180.0
    return mx, my


def gridsearch_best3(X,y, estimator, parameters, n_iter=10, test_size=0.3):
    cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=n_iter, test_size=test_size, random_state=0)
    clf = GridSearchCV(estimator, parameters, cv=cv, scoring=rel_rmse)
    gs = clf.fit(X,y)
    best3 = [gs.grid_scores_[i] for i in np.argsort([i.mean_validation_score for i in gs.grid_scores_])[:3]]
    return best3

def plot_best3(estimator, X, y, best3, ylim, n_iter=10, test_size=0.3):
    cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.3, random_state=0)
    fig = plt.figure(figsize=(15,3))
    c=1
    for params in best3:
        fig.add_subplot(1,len(best3),c)
        c+=1
        plot_learning_curve(estimator, str(params.parameters), X, y, ylim=ylim, scoring=rel_rmse, cv=cv, n_jobs=4)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scoring=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="train score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="test score")

    plt.legend(loc="best")

def plot_map(lat, lon, color=None, size=10):
    cmap = cm.rainbow
    wlat, wlong = latlng_to_meters(lat, lon)
    if color is not None:
        colors = MinMaxScaler(feature_range=(0,255)).fit_transform(color)
        colors = ["#%02x%02x%02x"%tuple([int(j*255) for j in cmap(int(i))[:3]]) for i in colors]

    openmap_url = 'http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png'
    otile_url = 'http://otile1.mqcdn.com/tiles/1.0.0/sat/{Z}/{X}/{Y}.jpg'

    TILES = WMTSTileSource(url=openmap_url)
    tools="pan,wheel_zoom,reset"
    p = figure(tools=tools, plot_width=700,plot_height=600)

    p.add_tile(TILES)

    p.axis.visible = False
    
    cb = figure(plot_width=40, plot_height=600,  tools=tools)
    yc = np.linspace(np.min(color),np.max(color),20)
    c = np.linspace(0,255,20).astype(int)
    dy = yc[1]-yc[0]    
    cb.rect(x=0.5, y=yc, color=["#%02x%02x%02x"%tuple([int(j*255) for j in cmap(int(i))[:3]]) for i in c], width=1, height = dy)
    cb.xaxis.visible = False
    p.circle(np.array(wlat), np.array(wlong), color=colors, size=size)
    pb = gridplot([[p, cb]])
    show(pb)
        
    
def rmse(estimator, X, y):
    preds = estimator.predict(X)
    return np.sqrt(np.mean((preds-y)**2))

def rel_rmse(estimator, X, y):
    preds = estimator.predict(X)
    return np.mean(np.abs(preds-y)/y)
