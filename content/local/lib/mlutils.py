from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import sys
import progressbar
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from time import time
#import tensorflow as tf
#from tensorflow.keras import backend as K
import time


classifiers = {
               0: [None, "None"],
               1: [LogisticRegression(solver="lbfgs"), "Linear Classifier"],
               2: [GaussianNB(), "Naive Gaussian"],
               3: [SVC(gamma=.5), "SVM gamma=0.1"], 
               4: [SVC(gamma=10), "SVM gamma=10"],
               5: [SVC(gamma=100), "SVM gamma=100"],
               6: [DecisionTreeClassifier(max_depth=2), "DecisionTree depth=2"],
               7: [DecisionTreeClassifier(max_depth=100), "DecisionTree depth=100"],
               8: [RandomForestClassifier(n_estimators=2, max_depth=2), "RandomForest 2 trees depth=2"],
               9: [RandomForestClassifier(n_estimators=20, max_depth=2), "RandomForest 20 trees depth=2"],
              10: [KNeighborsClassifier(n_neighbors=3), "3 neighbours"],
              11: [KNeighborsClassifier(n_neighbors=15), "15 neighbours"],
              }


def pbar(**kwargs):
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(.2)
    return progressbar.ProgressBar(**kwargs)

def xplot_2D_boundary(predict, mins, maxs, margin_pct=.2, n=200, line_width=3, 
                     line_color="black", line_alpha=1, line_style=None, label=None):
    n = 200 if n is None else n
    mins -= np.abs(mins)*margin_pct
    maxs += np.abs(maxs)*margin_pct
    d0 = np.linspace(mins[0], maxs[0],n)
    d1 = np.linspace(mins[1], maxs[1],n)
    gd0,gd1 = np.meshgrid(d0,d1)
    D = np.hstack((gd0.reshape(-1,1), gd1.reshape(-1,1)))
    preds = predict(D)
    levels = np.sort(np.unique(preds))
    levels = [np.min(levels)-1] + [np.mean(levels[i:i+2]) for i in range(len(levels)-1)] + [np.max(levels)+1]
    p = (preds*1.).reshape((n,n))
    plt.contour(gd0,gd1,p, levels=levels, alpha=line_alpha, colors=line_color, linestyles=line_style, linewidths=line_width)
    if label is not None:
        plt.plot([0,0],[0,0], lw=line_width, color=line_color, ls=line_style, label=label)
    return np.sum(p==0)*1./n**2, np.sum(p==1)*1./n**2

def plot_2D_boundary(predict, mins, maxs, margin_pct=.2, n=200, line_width=3, 
                     line_color="black", line_alpha=1, line_style=None, label=None,
                     background_cmap=None, background_alpha=.5):
    n = 200 if n is None else n
    mins -= np.abs(mins)*margin_pct
    maxs += np.abs(maxs)*margin_pct
    d0 = np.linspace(mins[0], maxs[0],n)
    d1 = np.linspace(mins[1], maxs[1],n)
    gd0,gd1 = np.meshgrid(d0,d1)
    D = np.hstack((gd0.reshape(-1,1), gd1.reshape(-1,1)))
    preds = predict(D)
    levels = np.sort(np.unique(preds))
    levels = [np.min(levels)-1] + [np.mean(levels[i:i+2]) for i in range(len(levels)-1)] + [np.max(levels)+1]
    p = (preds*1.).reshape((n,n))
    if background_cmap is not None:
        plt.contourf(gd0,gd1,p, levels=levels, cmap=background_cmap, alpha=background_alpha)
    plt.contour(gd0,gd1,p, levels=levels, colors=line_color, alpha=line_alpha, linestyles=line_style, linewidths=line_width)
    if label is not None:
        plt.plot([0,0],[0,0], lw=line_width, color=line_color, ls=line_style, label=label)
    return np.sum(p==0)*1./n**2, np.sum(p==1)*1./n**2


def plot_2Ddata_with_boundary(predict, X, y, line_width=3, line_alpha=1, line_color="black", dots_alpha=.5, label=None, noticks=False):
    mins,maxs = np.min(X,axis=0), np.max(X,axis=0)    
    plot_2Ddata(X,y,dots_alpha)
    p0, p1 = plot_2D_boundary(predict, mins, maxs, line_width=line_width, 
                line_color=line_color, line_alpha=line_alpha, label=label )
    if noticks:
        plt.xticks([])
        plt.yticks([])
        
    return p0, p1

def plot_contour(ax, X,Y,Z, xlabel=None, ylabel=None, 
                 cmap=None, title=None, alpha=.7,
                 contour_alpha = .5, levels=20,
                 plot_contour_lines=True,
                 plot_contour_labels=True, **kwargs):
    
    if plot_contour_lines:
        CS = ax.contour(X,Y,Z, levels=levels, alpha=contour_alpha, colors="k")
        if plot_contour_labels:
            ax.clabel(CS, inline=1, fontsize=10)
    ax.contourf(X,Y,Z, levels=levels, alpha=alpha, cmap=cmap)
    ax.grid(color="black", alpha=.3);
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    X,y = (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))
    
    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)
    return X,y

def plot_2Ddata(X, y, dots_alpha=.5, noticks=False):
    colors = cm.hsv(np.linspace(0, .7, len(np.unique(y))))
    for i, label in enumerate(np.unique(y)):
        plt.scatter(X[y==label][:,0], X[y==label][:,1], color=colors[i], alpha=dots_alpha)
    if noticks:
        plt.xticks([])
        plt.yticks([])


class Example_Bayes2DClassifier():
    
    def __init__ (self, mean0, cov0, mean1, cov1, w0=1, w1=1):
        self.rv0 = multivariate_normal(mean0, cov0)
        self.rv1 = multivariate_normal(mean1, cov1)
        self.w0  = w0
        self.w1  = w1

    def sample (self, n_samples=100):
        n = int(n_samples)
        n0 = int(n*1.*self.w0/(self.w0+self.w1))
        n1 = int(n) - n0
        X = np.vstack((self.rv0.rvs(n0), self.rv1.rvs(n1)))
        y = np.zeros(n)
        y[n0:] = 1
        
        return X,y
        
    def fit(self, X,y):
        pass
    
    def predict(self, X):
        p0 = self.rv0.pdf(X)
        p1 = self.rv1.pdf(X)
        return 1*(p1>p0)
    
    def score(self, X, y):
        return np.sum(self.predict(X)==y)*1./len(y)

    # get limits for numeric computation. 
    # points all along the bounding box should have very low probability
    def get_boundingbox_probs(self, pdf, box_size):
        lp = np.linspace(-box_size,box_size,50)
        cp = np.ones(len(lp))*lp[0]
        bp = np.sum([pdf([x,y]) for x,y in zip(lp, cp)]  + \
                    [pdf([x,y]) for x,y in zip(lp, -cp)] + \
                    [pdf([y,x]) for x,y in zip(lp, cp)]  + \
                    [pdf([y,x]) for x,y in zip(lp, -cp)])
        return bp
    
    def get_prob_mesh(self, xrng, yrng, n=100):
        rngs = np.exp(np.arange(15))
        for rng in rngs:
            bp0 = self.get_boundingbox_probs(self.rv0.pdf, rng)
            bp1 = self.get_boundingbox_probs(self.rv1.pdf, rng)
            if bp0<1e-1 and bp1<1e-1:
                break
        if rng==rngs[-1]:
            print ("warning: bounding box prob size",rng,"has prob",np.max([bp0, bp1]) )
        
        rng = 3
        
        # then, compute numerical approximation by building a grid
        mins, maxs = [-rng, -rng], [+rng, +rng]
        d0 = np.linspace(*xrng, num=n)
        d1 = np.linspace(*yrng, num=n)
        xmesh,ymesh = np.meshgrid(d0,d1)
        D = np.hstack((xmesh.reshape(-1,1), ymesh.reshape(-1,1)))

        p1 = np.r_[[self.rv1.pdf(i) for i in D]].reshape(n,n)
        p0 = np.r_[[self.rv0.pdf(i) for i in D]].reshape(n,n)

        return xmesh,ymesh,p0,p1
        
    def analytic_score(self):
        """
        returns the analytic score on the knowledge of the probability distributions.
        the computation is a numeric approximation.
        """



        rngs = np.exp(np.arange(15))
        for rng in rngs:
            bp0 = self.get_boundingbox_probs(self.rv0.pdf, rng)
            bp1 = self.get_boundingbox_probs(self.rv1.pdf, rng)
            if bp0<1e-9 and bp1<1e-9:
                break

        if rng==rngs[-1]:
            print ("warning: bounding box prob size",rng,"has prob",np.max([bp0, bp1]))
        
        # then, compute numerical approximation by building a grid
        mins, maxs = [-rng, -rng], [+rng, +rng]
        n = 100
        d0 = np.linspace(mins[0], maxs[0],n)
        d1 = np.linspace(mins[1], maxs[1],n)
        gd0,gd1 = np.meshgrid(d0,d1)
        D = np.hstack((gd0.reshape(-1,1), gd1.reshape(-1,1)))

        p1 = np.r_[[self.rv1.pdf(i) for i in D]]
        p0 = np.r_[[self.rv0.pdf(i) for i in D]]

        # grid points where distrib 1 has greater probability than distrib 0
        gx = (p1>p0)*1.

        # true positive and true negative rates
        tnr = np.sum(p0*(1-gx))/np.sum(p0)
        tpr = np.sum(p1*gx)/np.sum(p1)
        return (self.w0*tnr+self.w1*tpr)/(self.w0+self.w1)  

    def get_bayes_errors(self):
        minx0,maxx0 = -10,10
        minx1,maxx1 = -10,10
        xmesh, ymesh, p1,p2 = self.get_prob_mesh([minx0,maxx0],[minx1,maxx1])
        pmax= np.max(np.r_[[p1,p2]])
        ds = ((maxx0-minx0)/p1.shape[0])*((maxx1-minx1)/p1.shape[1])
        err1 = np.sum(p1[p2>p1]*ds)
        err2 = np.sum(p2[p1>p2]*ds)
        return err1, err2
    
    def plot_contours(self, fig=None, show_bayesians=False, resample_points=False):
        global xxx_sample_points
        X,_ = self.sample(n_samples=500)
        minx0,minx1 = 0,0
        maxx0,maxx1 = 5,5
        
        if resample_points or not "xxx_sample_points" in globals():
            print ("resampling points")
            s0,s1 = [], []
            s0 = np.random.random(5)*(maxx0-minx0)+minx0
            s1 = np.random.random(5)*(maxx1-minx1)+minx1
            xxx_sample_points = [s0,s1]            
        s0, s1 = xxx_sample_points
        labels = [chr(65+i) for i in range(len(s0))]

        e1,e2 = self.get_bayes_errors()
        
        if show_bayesians:
            err1 = ", bayes error = %.2f"%e1
            err2 = ", bayes error = %.2f"%e2
        else:
            err1, err2 =  "", ""
        
        
        
        xmesh, ymesh, p1,p2 = self.get_prob_mesh([minx0,maxx0],[minx1,maxx1])
        pmax= np.max(np.r_[[p1,p2]])
        if fig is None:
            fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(131)
        plt.xlim(minx0, maxx0); plt.ylim(minx1, maxx1)
        plot_contour(ax, xmesh, ymesh, p1, 
                     plot_contour_labels=False, plot_contour_lines=False,
                     contour_alpha=.5,
                     alpha=.7, cmap=plt.cm.Reds,
                     title="american trilobite"+err1, xlabel="trilobite size", ylabel="trilobite weight")
        plt.scatter(s0, s1, color="black", s=200, marker="+")
        for i in range(len(s0)):
            plt.text(s0[i]+(maxx0-minx0)*.05, s1[i], labels[i])
        if show_bayesians:
            plot_2D_boundary(self.predict, [minx0,minx1], [maxx0, maxx1], margin_pct=0.001, line_width=3, line_color="black")
        ax = fig.add_subplot(132)
        plt.xlim(minx0, maxx0); plt.ylim(minx1, maxx1)
        plot_contour(ax, xmesh, ymesh, p2, 
                     plot_contour_labels=False, plot_contour_lines=False,
                     contour_alpha=.5,
                     alpha=.7, cmap=plt.cm.Blues,
                     title="african trilobite"+err2, xlabel="trilobite size", ylabel="trilobite weight", vmin=0, vmax=pmax)
        plt.scatter(s0, s1, color="black", s=200, marker="+")
        if show_bayesians:
            plot_2D_boundary(self.predict, [minx0,minx1], [maxx0, maxx1], margin_pct=0.001, line_width=3, line_color="black")
        for i in range(len(s0)):
            plt.text(s0[i]+(maxx0-minx0)*.05, s1[i], labels[i])
        if show_bayesians:
            ax = fig.add_subplot(133)
            plt.xlim(minx0, maxx0); plt.ylim(minx1, maxx1)
            plot_contour(ax, xmesh, ymesh, (p1-p2), 
                         cmap=plt.cm.RdBu_r, levels=10, alpha=.3, 
                         title="NATURAL (bayesian) frontier", xlabel="trilobite size", ylabel="trilobite weight")
            plot_2D_boundary(self.predict, [minx0,minx1], [maxx0, maxx1], margin_pct=0.001, line_width=3, line_color="black")
            plt.scatter(s0, s1, color="black", s=200, marker="+")
            for i in range(len(s0)):
                plt.text(s0[i]+(maxx0-minx0)*.05, s1[i], labels[i])
        return fig

def display_distributions(x0,y0, s0, d0, x1, y1, s1, d1, dummy, show_bayesians=False):
    global do_resample_points
    if not "do_resample_points" in globals():
        do_resample_points = False
    mc = Example_Bayes2DClassifier(mean0=[x0, y0], cov0=[[s0, d0], [d0, s0+d0]],
                                           mean1=[x1, y1], cov1=[[s1, d1], [d1, s1+d1]])
    fig1 = mc.plot_contours(show_bayesians=show_bayesians, resample_points=do_resample_points)

def interact_distributions():
    from ipywidgets import FloatSlider, Label, GridBox, interactive, Layout, VBox, \
                           HBox, Checkbox, IntSlider, Box, Button, widgets
    fx0=FloatSlider(value=2, description=" ", min=.5, max=4., step=.2, continuous_update=False,
                    layout=Layout(width='auto', grid_area='vx0'))
    fy0=FloatSlider(value=3, description=" ", min=.5, max=4., step=.2, continuous_update=False,
                    layout=Layout(width='auto', grid_area='vy0'))
    fs0=FloatSlider(value=1, description=" ", min=.1, max=4., step=.2, continuous_update=False,
                    layout=Layout(width='auto', grid_area='vs0'))
    fd0=FloatSlider(value=.9, description=" ", min=-2., max=2., step=.1, continuous_update=False,
                    layout=Layout(width='auto', grid_area='vd0'))

    fx1=FloatSlider(value=2, description=" ", min=.5, max=4., step=.2, continuous_update=False,
                    layout=Layout(width='auto', grid_area='vx1'))
    fy1=FloatSlider(value=2, description=" ", min=.5, max=4., step=.2, continuous_update=False,
                    layout=Layout(width='auto', grid_area='vy1'))
    fs1=FloatSlider(value=1, description=" ", min=.1, max=4., step=.2, continuous_update=False,
                    layout=Layout(width='auto', grid_area='vs1'))
    fd1=FloatSlider(value=-.3, description=" ", min=-2., max=2., step=.1, continuous_update=False,
                    layout=Layout(width='auto', grid_area='vd1'))
    fdummy = FloatSlider(value=2, description= " ", min=1, max=4, step=1)
    
    l = lambda s,p, w="auto": Label(s, layout=Layout(width=w, grid_area=p))


    bay = Checkbox(value=False, description='show NATURAL frontiers',disabled=False, indent=False,
                   layout=Layout(width="80%"))

    resample = Button(description="resample data points")                

    from IPython.core.display import clear_output
    def resample_onclick(_):
        global do_resample_points
        do_resample_points = True
        tmp = fdummy.value
        fdummy.value = tmp+(1 if tmp<3 else -1)
        do_resample_points = False
        
    resample.on_click(resample_onclick)

    w = interactive(display_distributions,
                       x0=fx0, y0=fy0, s0=fs0, d0=fd0,
                       x1=fx1, y1=fy1, s1=fs1, d1=fd1, show_bayesians=bay, dummy=fdummy,
                       continuous_update=False)

    w.children[-1].layout=Layout(width='auto', grid_area='fig')

    controls = Box([bay, resample],
                     layout=Layout(grid_area="ctr", 
                     display="flex-flow",
                     justify_content="flex-start",
                     flex_flow="column",
                     align_items = 'flex-start'))

    gb =GridBox(children=[fx0, fy0, fs0, fd0, fx1, fy1, fs1, fd1,
                          l("AMERICAN TRILOBYTE", "h0"), l("AFRICAN TRILOBYTE", "h1"),
                          l("size", "lx0"),l("weight", "ly0"), l("spread", "ls0"), l("tilt", "ld0"),
                          l("size", "lx1"),l("weight", "ly1"), l("spread", "ls1"), l("tilt", "ld1"),
                          controls
                         ],
            layout=Layout(
                width='100%',
                grid_template_rows='auto auto auto auto auto auto auto',
                grid_template_columns='5% 30% 5% 30% 30%',
                grid_template_areas='''
                "h0 h0 h1 h1 ."
                "lx0 vx0 lx1 vx1 ."
                "ly0 vy0 ly1 vy1 ctr"
                "ls0 vs0 ls1 vs1 ctr"
                "ld0 vd0 ld1 vd1 ctr"
                "fig fig fig fig fig"
                ''')
           )


    def limit_fd0(*args):
        fd0.max = fs0.value+fs0.value*0.5
        fd0.min = -fs0.value*0.5
    def limit_fd1(*args):
        fd1.max = fs1.value+fs1.value*0.5
        fd1.min = -fs1.value*0.5
    fs0.observe(limit_fd0, "value")
    fd0.observe(limit_fd0, "value")
    fs1.observe(limit_fd1, "value")
    fd1.observe(limit_fd1, "value")

    w.children[0].value=1
    widget1 = VBox([gb, w.children[-1]])
    display(widget1)
    return fx0, fy0, fs0, fd0, fx1, fy1, fs1, fd1        
        
        
def display_traintest(n_samples, test_pct, show_bayesian, classifier):
    from time import time
    global params
    global classifiers
    x0,y0,s0,d0,x1,y1,s1,d1 = params
    minx0,minx1 = 0,0
    maxx0,maxx1 = 5,5

    mc = Example_Bayes2DClassifier(mean0=[x0.value, y0.value], cov0=[[s0.value, d0.value], [d0.value, s0.value+d0.value]],
                                           mean1=[x1.value, y1.value], cov1=[[s1.value, d1.value], [d1.value, s1.value+d1.value]])

    X,y = mc.sample(n_samples)
    global X_train, X_test, y_train, y_test, last_testpct, last_nsamples

    if not "last_testpct" in globals() or last_testpct!=test_pct or\
       not "last_nsamples" in globals() or last_nsamples!=n_samples:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_pct)
        last_testpct = test_pct
        last_nsamples = n_samples
    e0,e1 = mc.get_bayes_errors()
    berror_train0 = np.mean(mc.predict(X_train[y_train==0])!=0)
    berror_train1 = np.mean(mc.predict(X_train[y_train==1])!=1)
    berror_test0  = np.mean(mc.predict(X_test[y_test==0])!=0)
    berror_test1  = np.mean(mc.predict(X_test[y_test==1])!=1)
    est_label, time_str, train_str, test_str = "", "", "", ""
    if classifier is not None and classifier!=0:
        est = classifiers[classifier][0]
        est_label = classifiers[classifier][1]
        start = time()
        est.fit(X_train, y_train)
        fitting_time = time()-start
        time_str = "fit time  (TRAIN)    %.2f $\mu$secs"%(fitting_time*1000)
        
        start = time()
        test_score0 = np.mean(est.predict(X_test)[y_test==0]==0)
        test_score1 = np.mean(est.predict(X_test)[y_test==1]==1)
        predict_time = time()-start
        time_str += "\npredict time (TEST) %.2f $nano$secs"%(1000*predict_time)

        train_score0 = np.mean(est.predict(X_train)[y_train==0]==0)
        train_score1 = np.mean(est.predict(X_train)[y_train==1]==1)

        train_str = "errors:    reds %.1f%s  |   blues %.1f%s"%((1-train_score0)*100,"%", (1-train_score1)*100,"%")
        test_str  = "errors:    reds %.1f%s  |   blues %.1f%s"%((1-test_score0)*100,"%", (1-test_score1)*100,"%")

    def show_boundaries():
        if show_bayesian:
            plot_2D_boundary(mc.predict, [minx0,minx1], [maxx0, maxx1], margin_pct=0.001, 
                                     background_cmap=plt.cm.RdBu, background_alpha=.2,
                                     line_width=2, line_color="black", line_alpha=1, line_style="-")
        if classifier is not None and classifier!=0:
            plot_2D_boundary(est.predict, [minx0,minx1], [maxx0,maxx1], 
                                     background_cmap=plt.cm.RdBu, background_alpha=.2,
                                     line_width=1, line_color="black", line_alpha=1, line_style="--")

    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.scatter(X_train[:,0][y_train==0], X_train[:,1][y_train==0], color="red", alpha=.7)
    plt.scatter(X_train[:,0][y_train==1], X_train[:,1][y_train==1], color="blue", alpha=.7)
    show_boundaries()
    plt.title("TRAIN data, %d objects\n"%len(X_train)+train_str)
    plt.grid()
    plt.axis("equal")
    plt.xlim(0,5)
    plt.ylim(0,5)

    plt.subplot(132)
    plt.scatter(X_test[:,0][y_test==0], X_test[:,1][y_test==0], color="red", alpha=.7)
    plt.scatter(X_test[:,0][y_test==1], X_test[:,1][y_test==1], color="blue", alpha=.7)
    show_boundaries()
    plt.title("TEST data, %d objects\n"%len(X_test)+test_str)
    plt.grid()
    plt.axis("equal")
    plt.xlim(0,5)
    plt.ylim(0,5)

    plt.subplot(133)
    plt.text(0,  .8, est_label, fontsize=18)
    plt.text(0.2,.3, time_str, fontsize=14)
    if show_bayesian:
        plt.text(0,2, "Bayes (NATURAL) error", fontsize=18)
        plt.text(0,1.5,"reds", fontsize=14)
        plt.text(0,1.3,"blues", fontsize=14)
        plt.text(.5,1.5, "%.1f%s"%(e0*100,"%"), fontsize=14)
        plt.text(.5,1.3, "%.1f%s"%(e1*100,"%"), fontsize=14)
        plt.text(1,1.5, "%.1f%s"%(berror_train0*100,"%"), fontsize=14)
        plt.text(1,1.3, "%.1f%s"%(berror_train1*100,"%"), fontsize=14)
        plt.text(1.6,1.5, "%.1f%s"%(berror_test0*100,"%"), fontsize=14)
        plt.text(1.6,1.3, "%.1f%s"%(berror_test1*100,"%"), fontsize=14)
        plt.text(.4,1.73, "analytical", fontsize=12)
        plt.text(1,1.73, "TRAIN", fontsize=12)
        plt.text(1.6,1.73, "TEST", fontsize=12)
    plt.ylim(0,3)
    plt.xlim(0,2)
    plt.axis("off")

def interact_traintest(p):
    global params
    params = p
    from ipywidgets import FloatSlider, Label, GridBox, interactive, Layout, VBox, \
                           HBox, Checkbox, IntSlider, Box, Button, widgets, Dropdown

    fn_samples = IntSlider(value=101, description="# samples",  min=100, max=2000, step=100, continuous_update=False)
    ftest_pct =  FloatSlider(value=.25, description="% test",  min=.1, max=.9, step=.05, continuous_update=False)
    fbay =        Checkbox(value=False, description='show NATURAL frontiers',indent=False, continuous_update=False)

    cvals = [i[1] for i in classifiers.values()]    
    fclassifier = Dropdown(
        options=[[j[1],i] for i,j in classifiers.items()],
        value=0,
        description='Classifier:',
        disabled=False,
    )    

    fall = HBox([fn_samples, fclassifier, ftest_pct, fbay])

    w = interactive(display_traintest, n_samples=fn_samples, test_pct=ftest_pct, 
                    show_bayesian=fbay, classifier=fclassifier)
    w.children[0].value = 100
    widget1 = VBox([fall, w.children[-1]])
    display(widget1)      
        
def plot_estimator_border(bayes_classifier, estimator=None, 
                          mins=None, maxs=None,
                          estimator_name=None, X=None, y=None, n_samples=500,legend=True):    
    estimator_name = estimator.__class__.__name__ if estimator_name is None else estimator_name
    nns = [10,50,100]
    if X is None or y is None:
        X,y = bayes_classifier.sample(n_samples)
    mins = np.min(X, axis=0) if mins is None else mins
    maxs = np.max(X, axis=0) if maxs is None else maxs
    if estimator is not None:
        estimator.fit(X,y)
        plt.title(estimator_name+", estimator=%.3f"%estimator.score(X,y)+ "\nanalytic=%.3f"%bayes_classifier.analytic_score())
        plot_2D_boundary(estimator.predict, mins, maxs, 
                            line_width=1, line_alpha=.5, label="estimator boundaries")
    else:
        plt.title("analytic=%.3f"%bayes_classifier.analytic_score())
    plot_2Ddata(X, y, dots_alpha=.3)

    plot_2D_boundary(bayes_classifier.predict, mins, maxs, 
                             line_width=4, line_alpha=1., line_color="green", label="bayes boundary")

    plt.xlim(mins[0], maxs[0])
    plt.ylim(mins[1], maxs[1])

    if legend:
         plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def sample_borders(mc, estimator, samples, n_reps, mins=None, maxs=None):
    plt.figure(figsize=(15,3))
    for i,n_samples in pbar(max_value=len(samples))(enumerate(samples)):
        plt.subplot(1,len(samples),i+1)
        for ii in range(n_reps):
            X,y = mc.sample(n_samples)
            estimator.fit(X,y)
            if ii==0:
                plot_2D_boundary(estimator.predict, np.min(X, axis=0), np.max(X, axis=0), 
                                 line_width=1, line_alpha=.5, label="estimator boundaries")
            else:
                plot_2D_boundary(estimator.predict, np.min(X, axis=0), np.max(X, axis=0), 
                                 line_width=1, line_alpha=.5)                    
            plt.title("n samples="+str(n_samples))
        mins = np.min(X, axis=0) if mins is None else mins
        maxs = np.max(X, axis=0) if maxs is None else maxs
        plot_2D_boundary(mc.predict, mins, maxs, 
                         line_width=5, line_alpha=1., line_color="green", label="bayes boundary")
        plt.xlim(mins[0], maxs[0])
        plt.ylim(mins[1], maxs[1])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
from sklearn.neighbors import KernelDensity

class KDClassifier:
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def fit(self, X,y):
        """
        builds a kernel density estimator for each class
        """
        self.kdes = {}
        for c in np.unique(y):
            self.kdes[c] = KernelDensity(**self.kwargs)
            self.kdes[c].fit(X[y==c])
        return self
        
    def predict(self, X):
        """
        predicts the class with highest kernel density probability
        """
        classes = self.kdes.keys()
        preds = []
        for i in sorted(classes):
            preds.append(self.kdes[i].score_samples(X))
        preds = np.array(preds).T
        preds = preds.argmax(axis=1)
        preds = np.array([classes[i] for i in preds]) 
        return preds
    
    def score(self, X, y):
    
        return np.mean(y==self.predict(X))
    
    
def accuracy(y,preds):
    return np.mean(y==preds)

    
from sklearn.model_selection import train_test_split
def bootstrapcv(estimator, X,y, test_size, n_reps, score_func=None, score_funcs=None):

    if score_funcs is None and score_func is None:
        raise ValueError("must set score_func or score_funcs")
    
    if score_funcs is not None and score_func is not None:
        raise ValueError("cannot set both score_func and score_funcs")
    
    if score_func is not None:
        rtr, rts = [],[]
    else:
        rtr = {i.__name__:[] for i in score_funcs}
        rts = {i.__name__:[] for i in score_funcs}
        
    for i in range(n_reps):
        Xtr, Xts, ytr, yts = train_test_split(X,y,test_size=test_size)
        estimator.fit(Xtr, ytr)
        if score_func is not None:
            rts.append(score_func(yts, estimator.predict(Xts)))
            rtr.append(score_func(ytr, estimator.predict(Xtr)))
        else:
            for f in score_funcs:
                fname =  f.__name__
                rts[fname].append(f(yts, estimator.predict(Xts)))
                rtr[fname].append(f(ytr, estimator.predict(Xtr)))
    if score_func is not None:
        return np.array(rtr), np.array(rts)
    else:
        rtr = {i: np.array(rtr[i]) for i in rtr.keys()}
        rts = {i: np.array(rts[i]) for i in rts.keys()}
        return rtr, rts

def lcurve(estimator, X,y, n_reps, score_func, show_progress=False):
    test_sizes = np.linspace(.9,.1,9)
    trmeans, trstds, tsmeans, tsstds = [], [], [], []
    for test_size in pbar()(test_sizes):
        rtr, rts = bootstrapcv(estimator,X,y,test_size,n_reps, score_func)
        trmeans.append(np.mean(rtr))
        trstds.append(np.std(rtr))
        tsmeans.append(np.mean(rts))
        tsstds.append(np.std(rts))
    trmeans = np.array(trmeans)
    trstds  = np.array(trstds)
    tsmeans = np.array(tsmeans)
    trstds  = np.array(tsstds)
    abs_train_sizes = len(X)*(1-test_sizes)
    plt.plot(abs_train_sizes, trmeans, marker="o", color="red", label="train")
    plt.fill_between(abs_train_sizes, trmeans-trstds, trmeans+trstds, color="red", alpha=.2)
    plt.plot(abs_train_sizes, tsmeans, marker="o", color="green", label="test")
    plt.fill_between(abs_train_sizes, tsmeans-tsstds, tsmeans+tsstds, color="green", alpha=.2)
    plt.xlim(len(X)*.05, len(X)*.95)
    plt.xticks(abs_train_sizes)
    plt.grid()
    plt.xlabel("train size (%)")
    plt.ylabel(score_func.__name__)
    plt.ylim(0,1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
              ncol=2, fancybox=True, shadow=True)

def plot_cluster_predictions(clustering, X, n_clusters = None, cmap = plt.cm.plasma,
                             plot_data=True, plot_centers=True, show_metric=False,
                             title_str=""):

    assert not hasattr(clustering, "n_clusters") or \
           (hasattr(clustering, "n_clusters") and n_clusters is not None), "must specify `n_clusters` for "+str(clustering)

    if n_clusters is not None:
        clustering.n_clusters = n_clusters

    y = clustering.fit_predict(X)
    # remove elements tagged as noise (cluster nb<0)
    X = X[y>=0]
    y = y[y>=0]

    if n_clusters is None:
        n_clusters = len(np.unique(y))

    if plot_data:        
        plt.scatter(X[:,0], X[:,1], color=cmap((y*255./(n_clusters-1)).astype(int)), alpha=.5)
    if plot_centers and hasattr(clustering, "cluster_centers_"):
        plt.scatter(clustering.cluster_centers_[:,0], clustering.cluster_centers_[:,1], s=150,  lw=3,
                    facecolor=cmap((np.arange(n_clusters)*255./(n_clusters-1)).astype(int)),
                    edgecolor="black")   

    if show_metric:
        sc = silhouette_score(X, y) if len(np.unique(y))>1 else 0
        plt.title("n_clusters %d, sc=%.3f"%(n_clusters, sc)+title_str)
    else:
        plt.title("n_clusters %d"%n_clusters+title_str)

    plt.axis("off")
    return

def experiment_number_of_clusters(X, clustering, show_metric=True,
                                  plot_data=True, plot_centers=True, plot_boundaries=False):
    plt.figure(figsize=(15,6))
    for n_clusters in pbar()(range(2,10)):
        clustering.n_clusters = n_clusters
        y = clustering.fit_predict(X)

        cm = plt.cm.plasma
        plt.subplot(2,4,n_clusters-1)

        plot_cluster_predictions(clustering, X, n_clusters, cm, 
                                 plot_data, plot_centers, show_metric)


def experiment_KMeans_number_of_iterations(X, n_clusters=3,
                                    plot_data=True, plot_centers=True, plot_boundaries=False):
    plt.figure(figsize=(15,6))
    for i in pbar()(range(10)):
        init_centroids = np.vstack((np.linspace(np.min(X[:,0]), np.max(X[:,0])/20, n_clusters), 
                                    [np.min(X[:,1])]*n_clusters)).T

        x0min, x0max = np.min(X[:,0]), np.max(X[:,0])
        x1min, x1max = np.min(X[:,1]), np.max(X[:,1])
        c = np.random.random(size=(n_clusters, 2))/3
        c[:,0] = x0min + c[:,0]*(x0max-x0min)
        c[:,1] = x1min + c[:,1]*(x1max-x1min)
        init_centroids = c

        plt.subplot(2,5,i+1)
        cm = plt.cm.plasma
        
        if i==0:
            
            y = np.argmin(np.vstack([np.sqrt(np.sum((X-i)**2, axis=1)) for i in init_centroids]).T, axis=1)
            
            plt.scatter(X[:,0], X[:,1], color=cm((y*255./(n_clusters-1)).astype(int)), alpha=.5)
            plt.scatter(init_centroids[:,0], init_centroids[:,1], s=150,  lw=3,
                       facecolor=cm((np.arange(n_clusters)*255./(n_clusters-1)).astype(int)),
                       edgecolor="black")   
            plt.axis("off")
            plt.title("initial state")
            

        else:
            n_iterations = i if i<4 else (i-1)*2

            km = KMeans(n_clusters=n_clusters, init=init_centroids, n_init=1, max_iter=2*n_iterations)
            km.fit(X)

            plot_cluster_predictions(km, X, n_clusters, cm, plot_data, plot_centers, plot_boundaries)

            plt.title("n_iters %d"%(n_iterations))


def optimize(optimizer, loss, accuracy, params, test_mode):


    train_hist = []
    test_hist  = []

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        tf.tables_initializer().run()
        i=0
        while True:
            try:
                _, nloss,naccuracy = sess.run([optimizer, loss, accuracy])
                train_hist.append([nloss, naccuracy])
                if i%30==0:
                    test_nloss, test_naccuracy = sess.run([loss, accuracy], feed_dict={test_mode: True})
                    test_hist.append([test_nloss, test_naccuracy])
                    print ("\rstep %10d  train_acc %.2f test_acc %.2f"%(i,naccuracy, test_naccuracy),end="")
                i+=1
            except tf.errors.OutOfRangeError as e:
                print ("\nfinished iteration")
                break
        nparams = sess.run([params])
        train_hist, test_hist = np.r_[train_hist], np.r_[test_hist]
    return train_hist, test_hist, nparams

def logreg_model(train_input_fn, test_input_fn=None):
        
    test_input_fn = test_input_fn if test_input_fn is not None else train_input_fn
    
    # find out input size
    tf.reset_default_graph()
    nx,_ = test_input_fn()
    with tf.Session() as sess:
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()
        input_size = sess.run(nx).shape[1]    
    
    # now build the graph
    tf.reset_default_graph()
    train_nX, train_ny = train_input_fn()
    test_nX,  test_ny  = test_input_fn()

    test_mode = tf.Variable(initial_value=False, name="test_mode", dtype=tf.bool)
    next_X, next_y = tf.cond(test_mode, lambda: (test_nX, test_ny),
                                        lambda: (train_nX, train_ny)) 

    t = tf.Variable(initial_value=tf.random_uniform([input_size,1]), name="t", dtype=tf.float32)
    b = tf.Variable(initial_value=tf.random_uniform([1]), name="b", dtype=tf.float32)

    y_hat      = tf.sigmoid(b+tf.matmul(next_X,t))*.9+.05
    prediction = tf.reshape(tf.cast(y_hat>.5, tf.float32), (-1,1))
    accuracy   = tf.reduce_mean(tf.cast(tf.equal(prediction,next_y), tf.float32))
    
    loss = -tf.reduce_mean(next_y*tf.log(y_hat)+(1-next_y)*tf.log(1-y_hat))

    return y_hat, prediction, accuracy, loss, [t,b], test_mode



def plot_hists(train_hist, test_hist):

    plt.figure(figsize=(10,3))
    plt.subplot(121)
    plt.title("train loss")
    plt.grid()
    plt.plot(train_hist[:,0])
    plt.subplot(122)
    plt.plot(train_hist[:,1])
    plt.title("train accuracy")
    plt.grid()

    plt.figure(figsize=(10,3))
    plt.subplot(121)
    plt.title("test loss")
    plt.plot(test_hist[:,0])
    plt.grid()
    plt.subplot(122)
    plt.plot(test_hist[:,1])
    plt.title("test accuracy")
    plt.grid();

    
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size)/2. + (top + bottom)/2.
        for m in range(layer_size+(1 if n<layer_size else 0) ):
            color = "red" if n==0 else "blue" if n==len(layer_sizes)-1 else "gray"
            ec = "black"
            alpha = 1.
            if m==layer_size:
                ec = "gray"
                color = "white"
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color=color, ec=ec, zorder=4, alpha=alpha)
            ax.add_artist(circle)
            if m==layer_size:
                text = plt.Text(n*h_spacing + left - .015, layer_top - m*v_spacing - .015, "1", zorder=5)
                ax.add_artist(text)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b)/2. + (top + bottom)/2.
        for m in range(layer_size_a+1):
            for o in range(layer_size_b):
                color = "gray" if m==layer_size_a else "black"
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c=color, alpha=.5)
                ax.add_artist(line)
                
                
def display_imgs(w, figsize=(6,6)):
    plt.figure(figsize=figsize)
    w = (w-np.min(w))/(np.max(w)-np.min(w))
    for i in range(w.shape[-1]):
        plt.subplot(10,10,i+1)
        plt.imshow(w[:,:,:,i], interpolation="none")
        plt.axis("off")
        
def show_labeled_image_mosaic(imgs, labels, figsize=(12, 12), idxs=None):

    plt.figure(figsize=figsize)
    for labi,lab in [i for i in enumerate(np.unique(labels))]:
        k = imgs[labels == lab]
        _idxs = idxs[:10] if idxs is not None else np.random.permutation(len(k))[:10]
        for i, idx in enumerate(_idxs):
            if i == 0:
                plt.subplot(10, 11, labi*11+1)
                plt.title("LABEL %d" % lab)
                plt.plot(0, 0)
                plt.axis("off")

            img = k[idx]
            plt.subplot(10, 11, labi*11+i+2)
            plt.imshow(img, cmap=plt.cm.Greys_r)
            plt.axis("off")
            
            
def show_preds(x, y, preds):
    for i in range(len(x)):
        plt.figure(figsize=(5,2.5))
        plt.subplot(122)
        plt.imshow(x[i])
        plt.axis("off")
        plt.subplot(121)
        plt.bar(np.arange(len(preds[i])), preds[i], color="blue", alpha=.5, label="prediction")
        plt.bar(np.arange(len(preds[i])), np.eye(len(preds[i]))[int(y[i])], color="red", alpha=.5, label="label")
        plt.xticks(range(len(preds[i])), range(len(preds[i])), rotation="vertical");
        plt.xlim(-.5,len(preds[i])-.5);
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, +1.35),ncol=5)

        
def get_activations(model, model_inputs, layer_name=None):
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    # we remove the placeholders (Inputs node in Keras). Not the most elegant though..
    outputs = [output for output in outputs if 'input_' not in output.name]

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    activations = [func(list_inputs)[0] for func in funcs]
    layer_names = [output.name for output in outputs]

    result = dict(zip(layer_names, activations))
    return result


import ipywidgets as widgets
def make_form():
    k1 = widgets.Text(
        value='Hello World',
        placeholder='KA',
        description='Keyword A',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )

    k2 = widgets.Text(
        value='Hello Wrld',
        placeholder='KB',
        description='Keyword B',
        disabled=False, 
        layout=widgets.Layout(width='50%')
    )

    s = widgets.IntSlider(
        value=70,
        min=20,
        max=80,
        step=1,
        description='% for train:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    w = widgets.VBox((k1,k2,s))
    return w, k1, k2, s

def run_bash(cmd):
    print ("--> running", cmd)
    from IPython import get_ipython
    ipython = get_ipython()

    global OUT
    ipython.run_cell_magic("script", "bash --out OUT", cmd)
    print ("__>", OUT)
    return OUT.split("\n")

def search_google_build_dataset_stage1(k1, k2, train_pct, n_images, downloads, traintest_imgs):
    from IPython import get_ipython
    ipython = get_ipython()

    k1 = k1.replace(" ","-")
    k2 = k2.replace(" ","-")
    #!rm -rf $downloads
    #!mkdir $downloads
    run_bash("rm -rf %s"%downloads)
    run_bash("mkdir %s"%downloads)

    run_bash("rm -rf %s"%traintest_imgs)
    run_bash("mkdir %s"%traintest_imgs)
    
    cfg="""
    {
        "Records": [
            {
                "keywords": "%s",
                "format": "jpg",
                "limit": %d,
                "type": "photo",
                "size": "medium",
                "output_directory": "%s"
            },
            {
                "keywords": "%s",
                "format": "jpg",
                "limit": %d,
                "type": "photo",
                "size": "medium", 
                "output_directory": "%s"
            }
        ]
    }
    """%(k1, int(n_images), downloads, k2, int(n_images), downloads)
    with open(downloads+"/cfg.txt", "w") as f:
        f.write(cfg)
        
        
def search_google_build_dataset_stage2(k1, k2, train_pct, n_images, downloads, traintest_imgs):
    from IPython import get_ipython
    ipython = get_ipython()

    k1 = k1.replace(" ","-")
    k2 = k2.replace(" ","-")

    downloads = "/tmp/downloads"
    traintest_imgs = "/tmp/imgs"

    print ("-------------------")
    print ("cleaning images")

    from skimage import io
    files = run_bash("find %s -type f"%downloads)

    for fname in files:
        if fname!=downloads+"/cfg.txt" and fname.strip()!="":
            print (".", end="")
            try:
                io.imread(fname)
            except:
                print ("\nremoving",fname,"as cannot be read\n")
                run_bash("rm %s"%fname)

    print ("--------------------")
    print (" organizing images")

    #classdirs = !find $downloads -type d
    classdirs = run_bash("find %s -type d"%downloads)
    classdirs = [i.split("/")[-1] for i in classdirs if i!=downloads]
    print (classdirs)

    run_bash("rm -rf %s"%traintest_imgs)
    
    for classdir in classdirs:
        print (classdir)
        print ("    splitting files ... ", end=" ")
        files = run_bash("find $downloads/$classdir -type f"%(downloads, classdir))
        #files = !find $downloads/$classdir -type f
        files = np.r_[files]
        n_train = int(len(files)*train_pct)
        files = np.random.permutation(files)
        files_train = files[:n_train]
        files_test  = files[n_train:]
        #!mkdir -p $traintest_imgs/train/$classdir
        run_bash("mkdir -p %s/train/%s"%(traintest_imgs, classdir))
        #!mkdir -p $traintest_imgs/test/$classdir
        run_bash("mkdir -p %s/test/%s"%(traintest_imgs, classdir))
        print ("copying files")
        for f in files_train:
            #!cp $f $traintest_imgs/train/$classdir
            run_bash("cp %s %s/train/%s"%(f, traintest_imgs, classdir))
        for f in files_test:
            run_bash("cp %s %s/test/%s"%(f, traintest_imgs, classdir))
#            !cp $f $traintest_imgs/test/$classdir


    print ("--------------")
    print ("sanity check 1")

    for classdir in classdirs:
        print (classdir)
        #files = !find  $traintest_imgs/train/$classdir -type f
        files = run_bash("find  %s/train/%s -type f"%(traintest_imgs, classdir))
        print ("      ",len(files), "train images")
        #files = !find  $traintest_imgs/test/$classdir -type f
        files = run_bash("find  %s/test/%s -type f"%(traintest_imgs, classdir))
        print ("      ",len(files), "test images")


def figures_grid(nfigsx, nfigsy, figs_functions, figsize=None):
    if figsize is None:
        figsize = (nfigsx*3, nfigsy*3)
        
    fig = plt.figure(figsize=figsize)
    i = 1
    for y in range(1, nfigsy+1):
        for x in range(1,nfigsx+1):
            if i<=len(figs_functions):
                axis = fig.add_subplot(nfigsy, nfigsx, i)
                figs_functions[i-1]()
            i+=1

            
def show_1D_dataset_samples(n, d1, d2, n_datasets=10, dot_alpha=.5, line_alpha=.5, figsize=(20,5)):
    from sklearn.tree import DecisionTreeClassifier
    plt.figure(figsize=figsize)
    for i in range(n_datasets):

        m1 = d1.rvs(n)
        m2 = d2.rvs(n)
        X = np.append(m1, m2).reshape(-1,1)
        y = np.r_[[0]*len(m1)+[1]*len(m2)]
        estimator = DecisionTreeClassifier(max_depth=1)
        estimator.fit(X,y)
        Xr = np.linspace(5, 30, 100).reshape(-1,1)
        yr = estimator.predict(Xr)
        plt.plot(Xr[yr==0], [i]*np.sum(yr==0), color="red", alpha=line_alpha, lw=4)
        plt.plot(Xr[yr==1], [i]*np.sum(yr==1), color="blue", alpha=line_alpha, lw=4)
        plt.scatter(m1, [i+.1]*len(m1), color="red", alpha=dot_alpha, s=100)
        plt.scatter(m2, [i+.1]*len(m2), color="blue", alpha=dot_alpha, s=100)
    plt.axis("off")