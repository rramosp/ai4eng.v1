import sys
import os
import time
import progressbar
import numpy as np
import pickle
import pandas as pd
from joblib import Parallel, delayed


def timeseries_as_many2one(d, nb_timesteps_in, columns, timelag=0):
    t = {c: d[c].values for c in columns}
    X = []
    for i in range(len(d)-nb_timesteps_in-timelag):
        x = []
        for c in columns:
            x += list(t[c][i:i+nb_timesteps_in])
        X.append(x)
        
    colnames = []
    for c in columns:
        colnames += ["%s_%d"%(c, i) for i in range(nb_timesteps_in) ]

    X = np.r_[X].reshape(-1, nb_timesteps_in*len(columns))
    X = pd.DataFrame(X, index=d.index[nb_timesteps_in+timelag:], columns=colnames)
    r = X.join(d)
    return r

def lstm_as_many2one_timeseries_dataset(dl, nb_timestep_in,  target_column="target"):

    indices = []
    targets = []
    indices_target = []
    lstm_data = []

    nfolds = dl.shape[0]

    for i in pbar(maxval=nfolds)(range(dl.shape[0])):

        assert nb_timestep_in > 0, 'Error values loock'

        if nb_timestep_in+i <= dl.index.shape[0]:

            t_aux = dl.iloc[nb_timestep_in+i-1:nb_timestep_in+i]
            indices_target.append(t_aux.index.min())
            targets.append(t_aux[target_column].values)

            aux = dl[i:nb_timestep_in+i]
            _ = aux.pop(target_column)

            indices.append(aux.index.max())

            lstm_record = aux.values.reshape((nb_timestep_in,aux.shape[1]))

            columns = aux.columns.values

            if len(lstm_data)>0:
                lstm_data.append(lstm_record)
            else:
                lstm_data = [lstm_record]

    return np.r_[lstm_data], indices, np.array(targets), indices_target, columns


def to_timedelta(t):
    bd_class = pd.tseries.offsets.BusinessDay
    return t if type(t) in [bd_class, pd.Timedelta] else pd.Timedelta(t)

def pbar(**kwargs):
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(.2)
    return progressbar.ProgressBar(**kwargs)

class mParallel(Parallel):
    def _print(self, msg, msg_args):
        if self.verbose > 10:
           fmsg = '[%s]: %s' % (self, msg % msg_args)
           sys.stdout.write('\r ' + fmsg)
           sys.stdout.flush()

class Timeseries_Experiment:

    def __init__(self, data, train_period, test_period, metrics_funcs, metrics_funcs_args={},
                       gap_between_train_and_test = "1s", target_col="target",
                       predict_on_train=True, n_jobs=-1,
                       input_cols_to_results = [], ignore_columns=[],
                       as_many2one=False, nb_timesteps_in=None,
                       max_folds=None, metadata=None, description=None,
                       align_folds_to_weekstart = False,
                       target_mode = "vector",
                       loss_on_validation_data=False):

        assert as_many2one in [False, 'linearized', '3D'] , "invalid as_many2one only None, 'linearized' or '3D' allowed"
        assert target_mode in ["vector", "column", "onehot"] , "invalid target_mode only 'vector', 'column', 'onehot' allowed"
        assert type(data.index[0]) == pd.Timestamp, "data must be a time indexed dataframe"
        assert len(input_cols_to_results)==0 or np.alltrue([i in data.columns for i in input_cols_to_results]), "all input_cols_to_results must exist in data"
        assert metrics_funcs is not None, "must set metrics functions"
        assert not as_many2one or (nb_timesteps_in and nb_timesteps_in>0), "must set nb_timesteps_in>0 if using as_many2one"
        assert np.alltrue([i in metrics_funcs for i in metrics_funcs_args.keys()]), "function name in metrics_funcs_args not existing in metrics_funcs"


        self.estimator = None
        self.data      = data
        self.train_period = to_timedelta(train_period)
        self.test_period  = to_timedelta(test_period)
        self.gap_between_train_and_test = to_timedelta(gap_between_train_and_test)
        self.target_col   = target_col
        self.predict_on_train = predict_on_train
        self.metrics_funcs = metrics_funcs
        self.metrics_funcs_args = metrics_funcs_args
        self.input_cols_to_results = input_cols_to_results
        self.ignore_columns = ignore_columns
        self.n_jobs = n_jobs
        self.fold_results_test = {}
        self.fold_results_train = {}
        self.as_many2one = as_many2one
        self.nb_timesteps_in = nb_timesteps_in
        self.max_folds = max_folds
        self.metadata = metadata
        self.description = description if description is not None else "saved experiment"
        self.target_mode = target_mode
        self.loss_on_validation_data = loss_on_validation_data
        self.align_folds_to_weekstart = align_folds_to_weekstart
        # feature mode
        self.set_as_many2one()

        # target mode
        self.set_target_mode()

    def set_as_many2one(self):

        if self.as_many2one:

            dirname = "" if not "dir" in dir(self.data) else self.data.dir+"/"
            name    = "tseries" if not "name" in dir(self.data) else self.data.name
            self.m2o_pkl_fname = dirname + name +"_%d_timesteps_in.hd5"%self.nb_timesteps_in

            if os.path.isfile(self.m2o_pkl_fname):
                print ("using cached many2one dataset")
                d, di, t, ti, cols = pickle.load(open(self.m2o_pkl_fname, "rb"))

            else:
                print ("creating many2one dataset")
                d = self.data[[i for i in self.data.columns if i==self.target_col or i not in self.ignore_columns]]
                d, di, t, ti, cols  = lstm_as_many2one_timeseries_dataset(d,
                                                                            nb_timestep_in=self.nb_timesteps_in,
                                                                            target_column=self.target_col)
                self.m2o = (d,di,t,ti,cols)
                #pickle.dump((d,di,t,ti,cols), open(self.m2o_pkl_fname, "wb" ))

            assert len(di)==len(ti)==len(d)==len(t) and np.alltrue([di[i]==ti[i] for i in range(len(di))]), "error in many2one dataset generation"

            self.m2o_columns = cols

            if self.as_many2one == 'linearized':

                self.index = np.r_[di]
                self.X = d.reshape(-1, d.shape[1]*d.shape[2])

            if self.as_many2one == '3D':
                self.index = np.r_[di]
                self.X = d

        else:
            self.index = np.r_[[ pd.Timestamp(date) for date in self.data.index.values]]
            cols = [c for c in self.data.columns if c!=self.target_col and not c in self.ignore_columns]
            self.X =  self.data[cols].values

    def set_target_mode(self):

        if self.target_mode == 'vector':
            self.y = self.data.loc[self.index][self.target_col].values

        if self.target_mode == 'column':
            self.y = self.data.loc[self.index][[self.target_col]].values


        if self.target_mode == 'onehot':

            # assert 'target_class' == self.target_col, "a 'target_class' column is necessary"

            list_class = np.sort(np.unique(self.data[[self.target_col]].values))

            onehot_target = []
            for onehot in self.data.loc[self.index][self.target_col].values:
                onehot_target.append(1*(onehot==list_class))

            self.y = np.array(onehot_target)

    def set_estimator(self, estimator, fit_params={}):
        self.estimator = estimator
        self.fit_params = fit_params

    def get_fold_limits(self, test_start):
        test_start   = pd.Timestamp(test_start)
        test_end     = test_start + pd.Timedelta(self.test_period) - self.gap_between_train_and_test
        train_start  = test_start - pd.Timedelta(self.train_period)
        train_end    = test_start - self.gap_between_train_and_test

        # fix to mondays
        if train_start.weekday() ==5:
            train_start = train_start - pd.Timedelta('1d')
        elif train_start.weekday() ==6:
            train_start = train_start - pd.Timedelta('2d')
    
        return test_start, test_end, train_start, train_end

    def extract_train_test_data(self, dates):
        test_start, test_end, train_start, train_end = dates


        trix = np.r_[[(i>=train_start) & (i<=train_end) for i in self.index]]
        tsix = np.r_[[(i>=test_start) & (i<=test_end) for i in self.index]]
        Xtr, ytr = self.X[trix], self.y[trix]
        Xts, yts = self.X[tsix], self.y[tsix]
        tr_index = self.index[trix]
        ts_index = self.index[tsix]

        train_input_cols_to_results = self.data.loc[tr_index][[i for i in self.input_cols_to_results]]
        test_input_cols_to_results  = self.data.loc[ts_index][[i for i in self.input_cols_to_results]]

        return (Xtr, Xts, ytr, yts, tr_index, ts_index,
               train_input_cols_to_results, test_input_cols_to_results)

    def run_fold(self, test_start):

        assert self.estimator is not None, "must call set_estimator before running experiments"
        dates = self.get_fold_limits(test_start)

        test_start, test_end, train_start, train_end = dates
        # print('dates:')
        # print('train', train_start, train_end)
        # print('test', test_start, test_end)

        (Xtr, Xts, ytr, yts, tr_index, ts_index,
        train_input_cols_to_results, test_input_cols_to_results) = self.extract_train_test_data(dates)

        k = {i:test_input_cols_to_results[i].values for i in self.input_cols_to_results}

        if len(Xts)>0 and len(Xtr)>0:

            results_tr = Timeseries_Experiment_Resultset(metrics_funcs=self.metrics_funcs,
                                                     metrics_funcs_args=self.metrics_funcs_args,
                                                     extra_info_names=self.input_cols_to_results)

            results_ts = Timeseries_Experiment_Resultset(metrics_funcs=self.metrics_funcs,
                                                        metrics_funcs_args=self.metrics_funcs_args,
                                                        extra_info_names=self.input_cols_to_results)

            v = {"validation_data": (Xts, yts)} if self.loss_on_validation_data else {}
            self.estimator.fit(Xtr,ytr, **self.fit_params, **v)

            predsts = self.estimator.predict(Xts)
            # print(yts)
            # print(type(yts))
            tmp = yts#[:,0]

            if self.target_mode == 'column':
                predsts = predsts[:,0]
                tmp = yts[:,0]

            if self.target_mode == 'onehot':
                predsts = [(aux).argmax() for aux in predsts]
                tmp = [(aux).argmax() for aux in yts]

            probsts = {"probs": self.estimator.predict_proba(Xts)} if "predict_proba" in dir(self.estimator) else {}
            #print('probs', len(probsts))
            results_ts.ladd(ts_index, tmp, predsts, **probsts, **{i:test_input_cols_to_results[i].values for i in self.input_cols_to_results})
            results_ts.add_metainfo(test_start = test_start, test_end=test_end,
                                    train_start = train_start, train_end = train_end)
            if hasattr(self.estimator, "feature_importances_"):
                results_ts.add_metainfo(feature_importances=self.estimator.feature_importances_)

            if self.predict_on_train:
                predstr = self.estimator.predict(Xtr)
                predstr = predstr[:,0] if len(predstr.shape)==2 else predstr
                probstr = {"probs": self.estimator.predict_proba(Xtr)} if "predict_proba" in dir(self.estimator) else {}
                tmp = ytr[:,0] if len(ytr.shape)==2 else ytr
                results_tr.ladd(tr_index, tmp, predstr, **probstr, **{i:train_input_cols_to_results[i].values for i in self.input_cols_to_results})
                results_tr.add_metainfo(test_start = test_start, test_end=test_end,
                                        train_start = train_start, train_end = train_end)
                if hasattr(self.estimator, "feature_importances_"):
                    results_tr.add_metainfo(feature_importances=self.estimator.feature_importances_)

            results_ts.close()
            results_tr.close()

            return results_ts, results_tr

        else:
            
            return None

    def get_folds_info(self, test_start=None, test_end=None):
        test_start = pd.Timestamp(test_start) if test_start is not None else None
        test_end   = pd.Timestamp(test_end) if test_end is not None else None
        test_start = np.min(self.data.index) + self.train_period if test_start is None else test_start
        test_start = pd.Timestamp(test_start)

        test_end = np.max(self.data.index) if test_end is None else test_end
        test_end = pd.Timestamp(test_end)

        assert test_end>test_start, "test_start %s must be before test_end %s"%(str(test_start), str(test_end))

        self.fold_results_test = {}
        self.fold_results_train = {}
        r = []
        n_folds = 0
        ftest_start = test_start
        while (ftest_start<=test_end):
            ftest_start, ftest_end, ftrain_start, ftrain_end = self.get_fold_limits(ftest_start)

            # fix to mondays
            if self.align_folds_to_weekstart and ftrain_start.weekday() == 5:
                ftrain_start = ftrain_start - pd.Timedelta('1d')
            elif self.align_folds_to_weekstart and ftrain_start.weekday() ==6:
                ftrain_start = ftrain_start - pd.Timedelta('2d')

            if not self.align_folds_to_weekstart or ftest_start.weekday() <= 4:
                r.append( {"test_start": ftest_start, "test_end": ftest_end,
                        "train_start": ftrain_start, "train_end": ftrain_end})
                
                n_folds += 1
            ftest_start += self.test_period
        return r

    def print_folds_info(self, date_fmt="%Y-%m-%d %H:%M"):
        f = self.get_folds_info()
        print ("experiment has %d time based folds"%len(f))
        print ("------------------------------------")
        print ("train start           train end             test start            test end")
        r = [i["train_start"].strftime(date_fmt)+"  --  "+i["train_end"].strftime(date_fmt)+"      " +\
            i["test_start"].strftime(date_fmt)+"  --  "+i["test_end"].strftime(date_fmt) for i in f]
        print ("\n".join(r))


    def run(self, test_start=None, test_end=None):
        from time import time
        start_t = time()

        from joblib import delayed
        folds_info = self.get_folds_info(test_start, test_end)
        folds_info = folds_info[:self.max_folds] if self.max_folds else folds_info

        # now run the folds
        if self.n_jobs==1:
            for fold_info in pbar()(folds_info):
                #print('test_start', fold_info["test_start"])
                resu = self.run_fold(fold_info["test_start"])

                if resu is not None:
                    results_ts, results_tr = resu
                    self.fold_results_test[fold_info["test_start"]] = results_ts
                    self.fold_results_train[fold_info["test_start"]] = results_tr
        else:
            #print(folds_info)
            f = lambda x: (x["test_start"], self.run_fold(x["test_start"]))
            r = mParallel(n_jobs=self.n_jobs, verbose=30)(delayed(f)(i) for i in folds_info)
            for test_start, resu in r:
                if resu is not None:
                    results_ts, results_tr = resu
                    self.fold_results_test[test_start] = results_ts
                    self.fold_results_train[test_start] = results_tr

        self.results_test = None
        for v in self.fold_results_test.values():
            self.results_test = v if self.results_test is None else self.results_test.append(v)

        self.results_train = None
        for v in self.fold_results_train.values():
            self.results_train = v if self.results_train is None else self.results_train.append(v)

        # if self.autosave_dir is not None:
        #     self.save(self.autosave_dir)

        self.run_time = time()-start_t

    def save(self, dir_name):
        import pickle, datetime
        r = {i: self.__getattribute__(i) for i in self.__dict__ if not i in ["data"]}

        from copy import copy
        dr = {}
        dr["data_start_date"] = np.min(self.data.index)
        dr["data_end_date"] = np.max(self.data.index)
        dr["data_len"] = len(self.data)
        dr["data_columns"] = self.data.columns


        r = copy(self)
        r.data = dr
        r.fold_results_test = None
        r.fold_results_train = None
        now = str(datetime.datetime.now()).replace(" ", "__")

        fname="%s/%s_%s_%d.pkl.bz"%(dir_name, self.estimator.__class__.__name__, str(now).split(".")[0], id(self))
        import bz2, pickle
        pickle.dump(r, bz2.BZ2File(fname, "w"))
        print ("\nexperiment config saved to", fname)

    @staticmethod
    def load(fname, with_data=None):
        import bz2, pickle
        r = pickle.load(bz2.BZ2File(fname, "r"))
        if with_data is not None:
            data_spec = r.data
            r.data = with_data.loc[data_spec["data_start_date"]:data_spec["data_end_date"]][data_spec["data_columns"]]
        return r

def fix_outrange_price_predictions(results):
    results["pred"] = [i for i in map(lambda x: 0 if x<0 else 180 if x>180 else x,results.pred.values)]
    return results

def filter_outrange_price_predictions(results):
    return results[(results.pred>=0)&(results.pred<=180)&(results.target!=0)].copy()

class Timeseries_Experiment_Resultset:

    def __init__(self, metrics_funcs, metrics_funcs_args={}, extra_info_names = []):
        """
        extra_info_names: variable names for extra info at each result report
        metrics_funcs: set of functions to be called upon get_metrics below on resampled result dataframes
                       holding at least "target" and "pred" columns. If "binary", automatically include
                       metrics for binary classification.
        """
        self.dates   = []
        self.targets = []
        self.preds   = []
        self.probs   = []
        self.extra_info = {i:[] for i in extra_info_names}
        self.is_closed = False
        self.metainfo = {}
        self.metrics_funcs = ["count"]+metrics_funcs if not "count" in metrics_funcs else metrics_funcs
        self.metrics_funcs_args = metrics_funcs_args


    def add(self, date, target, pred, **kwargs):
        assert not self.is_closed, "this resultset has already been closed"
        self.dates.append(date)
        self.targets.append(target)
        self.preds.append(pred)

        if "probs" in kwargs.keys():
            self.probs.append(kwargs["probs"])

        for k in self.extra_info.keys():
            assert k in kwargs.keys(), "extra info %s not reported for date %s"%(k, str(date))
            self.extra_info[k].append(kwargs[k])

    def ladd(self, dates, targets, preds, **kwargs):
        dates = list(dates)
        targets = list(targets)
        preds = list(preds)
        assert not self.is_closed, "this resultset has already been closed"
        n = len(dates)
        assert len(targets)==n and len(preds)==n, "all lists must have the same number of items"
        self.dates += list(dates)
        self.targets += list(targets)
        self.preds += list(preds)

        if "probs" in kwargs.keys():
            self.probs += list(kwargs["probs"])

        for k in self.extra_info.keys():
            assert k in kwargs.keys(), "extra info %s not reported for date %s"%(k, str(date))
            tmp = list(kwargs[k])
            assert len(tmp)==n,  "all lists must have the same number of items"
            self.extra_info[k] += tmp

    def add_metainfo(self, **kwargs):
        for k,v in kwargs.items():
            self.metainfo[k] = v

    def close(self):

        r = pd.DataFrame(np.r_[[self.targets, self.preds] + list(self.extra_info.values())].T,
                         index = self.dates,
                         columns = ["target", "pred"] + list(self.extra_info.keys()))

        if len(self.probs)>0:
            r["probs"] = self.probs
        self.details = r
        self.is_closed = True

        # check if is necesary save the dataset

    def get_metrics(self, groupby=None, resampling_period=None):
        assert self.metrics_funcs is not None, "must set metrics functions for this experiment"
        a = 1 if groupby else 0
        b = 1 if resampling_period else 0
        assert a+b!=2, "cannot set both groupby and resampling period"

        # get all data if no grouping or sampling
        if a+b==0:
            resampling_period = pd.Timedelta(np.max(self.details.index) - np.min(self.details.index)) + pd.Timedelta("1s")

        g=pd.Grouper(freq=resampling_period) if resampling_period else groupby

        r = None
        for fname in self.metrics_funcs:
            fargs = {} if fname not in self.metrics_funcs_args.keys() else self.metrics_funcs_args[fname]
            fname = "metrics_"+fname
            f = lambda x: self.__getattribute__(fname)(x, **fargs)
            k = self.details.groupby(g).apply(f)
            r = k if r is None else r.join(k)

        return r

    def append(self, other):
        assert self.is_closed and other.is_closed, "resultsets must be closed"

        # print(self.details.columns)
        # print(other.details.columns)

        assert len(self.details.columns)==len(other.details.columns) and \
               np.alltrue([self.details.columns[i]==other.details.columns[i] for i in range(len(self.details.columns))]), \
               "result sets must have the same column structre"

        r = self.__class__(metrics_funcs = self.metrics_funcs, metrics_funcs_args=self.metrics_funcs_args, extra_info_names=list(self.extra_info.keys()))
        r.details = self.details.append(other.details)
        r.is_closed = True
        return r

    def plot(self, **fig_kwargs):
        from bokeh.plotting import figure, show
        k = self.details
        bfig = figure(y_axis_label='price', x_axis_type='datetime', **fig_kwargs)
        bfig.line(k.index, k.target, color="navy", line_width=2, legend="target", alpha=.5)
        bfig.line(k.index, k.pred, color="red", line_width=2, legend="prediction", alpha=.5)
        show(bfig)


    @staticmethod
    def metrics_binary(x):
        y = x.target
        p = x.pred
        acc = np.mean(y==p) if len(y)>0 else 0
        tpr = np.mean(p[y==1]) if sum(y==1)>0 else 1
        tnr = np.mean(1-p[y==0]) if sum(y==0)>0 else 1
        fpr = np.mean(p[y==0]) if sum(y==0)>0 else 1
        fnr = np.mean(1-p[y==1]) if sum(y==1)>0 else 1
        return pd.Series([acc,tpr, fnr, tnr, fpr], index=["accuracy", "tpr", "fnr", "tnr", "fpr"])


    @staticmethod
    def metrics_multiclass_ignore_nones(x):
        y = x.target.values
        p = x.pred.values
        acc = np.mean( y[p!=None]==p[p!=None])
        pct = np.mean(p!=None)
        return pd.Series([acc, pct], index=["accuracy", "pct_predicted"])

    @staticmethod
    def metrics_n_classes(x, class_labels):
        y = x.target
        p = x.pred
        global_acc = np.mean(y==p) if len(y)>0 else 0
        class_prec = [np.mean(y[p==i]==i) for i in class_labels]
        class_rec  = [np.mean(p[y==i]==i) for i in class_labels]
        return pd.Series([global_acc]+\
                        class_prec+class_rec,
                        index=["global_acc"]+\
                            ["%d_prec"%i for i in class_labels]+\
                            ["%d_recall"%i for i in class_labels]).sort_index()


    @staticmethod
    def metrics_mape(x):
        y = x.target
        p = x.pred
        mape = np.mean(np.abs(y-p)/np.mean(y))
        return pd.Series([mape], index=["mape"])

    @staticmethod
    def metrics_trend(x, include_class_distribution=False):
        y = x.target
        p = x.pred
        gt = np.mean(p[y>0]>0)
        lt = np.mean(p[y<0]<0)
        eq = np.mean(p[y==0]==0)
        if include_class_distribution:
            gtd = np.mean(y>0)
            ltd = np.mean(y<0)
            eqd = np.mean(y==0)
            return pd.Series([lt,eq,gt, ltd, eqd, gtd],
                              index=["<0", "=0", ">0", "<0(%)", "=0(%)", ">0(%)"])
        else:
            return pd.Series([lt,eq,gt], index=["<0", "=0", ">0"])

    @staticmethod
    def metrics_count(x):
        return pd.Series([len(x)], index=["count"])

    @staticmethod
    def metrics_rmse(x):
        y = x.target.values
        p = x.pred.values
        return pd.Series([np.sqrt(np.mean((y-p)**2))], index=["rmse"])

    @staticmethod
    def metrics_pnlexpectation(x, L0_value=0):

        pred = x.pred.values
        y    = x.delta_price.values

        y_dn = y[pred<L0_value]
        #p_dn = pred[pred<L0_value]

        y_up = y[pred>L0_value]
        #p_up = pred[pred>L0_value]

        eloss_dn, eprof_dn = (np.mean(y_dn[y_dn>0])*np.mean(y_dn>0), -np.mean(y_dn[y_dn<0])*np.mean(y_dn<0)) if np.sum(y_dn>0)>0 else (0.,0.)
        eloss_up, eprof_up = (-np.mean(y_up[y_up<0])*np.mean(y_up<0), np.mean(y_up[y_up>0])*np.mean(y_up>0)) if np.sum(y_up<0)>0 else (0.,0.)
        epnl_dn = eprof_dn - eloss_dn
        epnl_up = eprof_up - eloss_up
        eprof = eprof_dn*np.mean(pred<L0_value) + eprof_up*np.mean(pred>L0_value)
        eloss = eloss_dn*np.mean(pred<L0_value) + eloss_up*np.mean(pred>L0_value)

        dpup = np.mean(pred > L0_value)
        dpdn = np.mean(pred < L0_value)
        dyup = np.mean(y > 0)
        dydn = np.mean(y < 0)
        acc_dn = np.mean(pred[y < 0] < L0_value)
        acc_up = np.mean(pred[y > 0]>L0_value)
        acc_zr = np.mean(pred[y==0]==L0_value)


        return pd.Series([eloss_up, eloss_dn, eprof_up, eprof_dn,
                           epnl_up, epnl_dn, eprof, eloss, eprof-eloss, eprof/eloss,
                           dpup, dpdn, dyup, dydn, acc_dn, acc_up, acc_zr],
                    index=["E_loss_Lp+","E_loss_Lp-","E_profit_Lp+","E_profit_Lp-",
                           "PNL_Lp+", "PNL_Lp-", "E_profit", "E_loss", "PNL", "PL_rate",
                           "P(Lp+)", "P(Lp-)","P(y+)", "P(y-)", "acc-", "acc+", "acc_0"
                           ])

    @staticmethod
    def metrics_riskprofit(x, class_spec=None, n_classes=None):
        """
        class_spec: i.e. {"-":[0,1], "0":[2,3]  "+": [4,5,6]} details what classes stand for
                    a positive/negative/zero price difference
                    if None, n_classes must be given and be of odd length so that the center
                    class is taken as "0", and above/below as positive/negative. For instance,
                    n_classes=5 results in {"-":[0,1], "0":[2]  "+": [3,4]}
        """
        assert n_classes is not None or class_spec is not None, "must set n_classes or class_spec"
        if class_spec is None:
            assert n_classes%2==1, "n_classes must be odd so that the center class is considered the zero class"
            classes_zero = [int(n_classes/2)]
            classes_negative = [i for i in range(classes_zero[0])]
            classes_positive = [i for i in range(classes_zero[0]+1, n_classes)]

            class_spec = {"-":classes_negative, "0": classes_zero, "+": classes_positive}

        z  = class_spec["0"]
        po = class_spec["+"]
        no = class_spec["-"]
        zp = z + po
        zn = z + no

        p = x.pred.values
        valid_preds = np.logical_not(pd.isna(p))
        valid_preds_pct = np.mean(valid_preds)
        p = x.pred.values[valid_preds]
        y = x.target.values[valid_preds]

        # risk free
        rfree = np.mean([ (y[i] in zp and p[i] in zp) or (y[i] in zn and p[i] in zn) for i in range(len(y))])

        # profit only
        ponly = np.mean([ ((y[i] in po and p[i] in po) or \
                           (y[i] in no and p[i] in no)) for i in range(len(y)) if p[i] not in z and p[i] not in z])

        puponly = np.mean([ (y[i] in po) for i in range(len(y)) if p[i] in po])
        pdnonly = np.mean([ (y[i] in no) for i in range(len(y)) if p[i] in no])

        lonly = np.mean([ ((y[i] in po and p[i] in no) or \
                           (y[i] in no and p[i] in po)) for i in range(len(y)) if p[i] not in z])

        ldnonly = np.mean([ (y[i] in po) for i in range(len(y)) if p[i] in no])
        luponly = np.mean([ (y[i] in no) for i in range(len(y)) if p[i] in po])

        return pd.Series([rfree, ponly, pdnonly, puponly, lonly, ldnonly, luponly, valid_preds_pct],
                          index=["risk_free_accuracy",
                                 "profit_only_accuracy", "profitdn_only_accuracy", "profitup_only_accuracy",
                                 "loss_only_accuracy", "lossdn_only_accuracy", "lossup_only_accuracy",
                                 "prediction_pct    "])
