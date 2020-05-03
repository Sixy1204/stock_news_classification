# A classifier for bda_mid_project by skearn  
Goal: Use a classifier to predict the stock price ups(+1) and downs(-1) during a period of time based on stock news.  

##From decision tree to XGBoost
1. Build decision tree by using default setting in sklearn
```python
tree.DecisionTreeClassifier() 
```
2. Build xgboost
- Search optimal parameter in parameter space
```python
# XGB parameters
xgb_clf_params = {
    'learning_rate':    hp.uniform('learning_rate', 0.001, 0.01),
    'max_depth':        scope.int(hp.quniform('max_depth', 5, 20, q=1)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     scope.int(hp.quniform('n_estimators', 50, 300, q=1))
}
xgb_fit_params = {
    'eval_metric': 'logloss',
    'early_stopping_rounds': 10,
    'verbose': False
}

xgb_para = dict()
xgb_para['clf_params'] = xgb_clf_params
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func' ] = lambda y, pred: log_loss(y, pred, eps=1e-15, normalize=True)
```
```python
class XGopt(object):
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        return result, trials

    def xgb_clf(self, para):
        clf = XGBClassifier(**para['clf_params'])
        return self.train_clf(clf, para)

    def train_clf(self, clf, para):
        clf.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = clf.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}
```
![GitHub](https://github.com/Sixy1204/stock_news_classification/tree/master/images "hyperopt")


