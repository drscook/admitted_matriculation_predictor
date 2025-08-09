exec(open('./students.py').read())
from sklearn.metrics import log_loss
def custom_log_loss(X_val, y_val, estimator, labels, X_train, y_train, weight_val=None, weight_train=None, config=None, groups_val=None, groups_train=None):
    """Some (crse,styp) are entirely False which causes an error with built-in log_loss. We create a custom_log_loss simply to set labels=[False, True] https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/"""
    start = time.time()
    y_pred = estimator.predict_proba(X_val)
    pred_time = (time.time() - start) / len(X_val)
    val_loss = log_loss(y_val, y_pred, labels=[False,True], sample_weight=weight_val)
    y_pred = estimator.predict_proba(X_train)
    train_loss = log_loss(y_train, y_pred, labels=[False,True], sample_weight=weight_train)
    return val_loss, {"val_loss": val_loss, "train_loss": train_loss, "pred_time": pred_time}

class Term(Core):
    def __init__(
        self,
        is_learner=True,
        features=dict(),
        subpops='styp_code',
        aggregates=None,
        flaml=dict(),
        **kwargs):
        super().__init__(**kwargs)
        self.is_learner = is_learner
        self.features = features
        self.subpops = union(subpops)
        self.aggregates = difference(union('crse_code', aggregates), subpops)
        self.flaml = flaml
        self.current = Students(**kwargs)
        kwargs.pop('date')
        self.stable = Students(date=self.stable_date, **kwargs)
        self.crse_code = '_headcnt'


    def get_enrollments(self):
        def fcn():
            def fcn1(agg):
                grp = union('crse_code', self.subpops, agg)
                g = lambda X, Y: get_incoming(X).join(Y, how='left', rsuffix='_y').groupmy(grp)['count'].sum()  # get stuff from Y that is not in X
                df = pd.DataFrame({
                    'current':g(self.current.get_students(), self.stable.get_registrations()),
                    'actual' :g(self.stable.get_registrations(), self.stable.get_students()),
                    }).fillna(0)
                df['mlt'] = df['actual'] / df['current']
                return df.sort_index()
            return {agg: fcn1(agg) for agg in self.aggregates}
        return self.run(fcn, f'enrollments/{self.date}/{self.term_code}', [self.current.get_students, self.stable.get_students, self.stable.get_registrations], suffix='.pkl')[0]


    def get_imputed(self):
        def fcn():
            def fcn1(df):
                X = df.fillna(self.features)[self.features.keys()].prep(category=True)
                imp = mf.ImputationKernel(X.reset_index(drop=True), random_state=self.seed)
                imp.mice(10)
                return imp.complete_data().set_index(X.index)
            return {s: fcn1(df) for s, df in self.current.get_students().groupmy(self.subpops)}
        return self.run(fcn, f'imputed/{self.date}/{self.term_code}', self.current.get_students, suffix='.pkl')[0]


    def get_prepared(self):
        def fcn(X):
            g = lambda k, v: self[k].get_registrations().query(f"crse_code==@v")['count'].rename(k if v==self.crse_code else v)
            Z = (X
                .join(g('current', '_tot_sch'))
                .join(g('current', self.crse_code).astype('boolean'))
                .join(g('stable' , self.crse_code).astype('boolean'))
                .fillna({'_tot_sch':0, 'current':False, 'stable':False})
            )
            if self.crse_code == '_proba':
                Z = Z.drop(columns=union(races, 'gender', 'international'), errors='ignore')
            return [Z, Z.pop('stable')]
        return {s: fcn(X) for s, X in self.get_imputed().items()}


    def get_learners(self):
        """train model - biggest bottleneck - can we run multiple (crse_code, year) in parallel?"""
        assert self.is_learner
        def fcn():
            def fcn1(s, Z):
                dct = {
                    'time_budget':30,
                    # 'max_iter': 100,
                    'task':'classification',
                    # 'log_file_name': self.get_dst(f'learners/{self.date}/{self.term_code}/{crse_code}/{s}', suffix='.log')[1],
                    # 'log_training_metric':True,
                    # 'log_type': 'all',
                    # 'log_training_metric':True,
                    'verbose':0,
                    'metric':custom_log_loss,
                    'eval_method':'cv',
                    'n_splits':3,
                    'seed':self.seed,
                    # 'early_stop':True,
                    'estimator_list': ['xgboost'],
                } | self.flaml
                learner = fl.AutoML(**dct)
                learner.fit(*Z, **dct)
                return learner
            return {s: fcn1(s, Z) for s, Z in self.get_prepared().items()}
        return self.run(fcn, f'learners/{self.date}/{self.term_code}/{self.crse_code}', [self.get_imputed, self.current.get_registrations, self.stable.get_registrations], suffix='.pkl', keep=False)[0]


    def get_predictions(self, model=None):
        model = (model or self).set('crse_code', self.crse_code)
        def fcn():
            data = self.get_prepared()
            learners = model.get_learners()
            dct = {s: pd.DataFrame({'prediction': learners[s].predict_proba(X).T[1], 'actual':y}) for s, [X, y] in data.items()}
            return (
                pd.concat(dct, names=union(self.subpops, 'pidm'))
                .assign(error=lambda X: X['prediction'] - X['actual'])
                .join(model.get_enrollments()['crse_code'].loc[model.crse_code, 'mlt'])
            )
            # L = [pd.DataFrame({'prediction': learners[subpop].predict_proba(X).T[1], 'actual':y}) for subpop, [X, y] in data.items()]
            # pred = pd.concat(L).assign(error=lambda X: X['prediction'] - X['actual'])
            return pred
        return self.run(fcn, f'predictions/{self.date}/{self.term_code}/{self.crse_code}/{model.term_code}', [self.get_prepared, model.get_learners], keep=False)[0]