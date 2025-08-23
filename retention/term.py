exec(open('./students.py').read())
import miceforest as mf, flaml as fl
from sklearn.metrics import log_loss

def custom_log_loss(X_val, y_val, estimator, labels, X_train, y_train, *args, **kwargs):
    """Flaml's log_loss errs when only 1 value appears in a split, so we create custom_log_loss to specify labels: https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/"""
    #I think there is an easier solution, but I can't find it.
    val_loss = log_loss(y_val, estimator.predict_proba(X_val), labels=[False,True])
    train_loss = log_loss(y_train, estimator.predict_proba(X_train), labels=[False,True])
    return val_loss, {"val_loss": val_loss, "train_loss": train_loss}

class Term(Core):
    def __init__(self, is_learner=True, features=dict(), subpops='styp_desc', aggregates=None, flaml=dict(), lag=0, **kwargs):
        super().__init__(**kwargs)
        self.is_learner = is_learner
        self.features = features
        self.subpops = union(subpops)
        self.aggregates = difference(union('crse_code', aggregates), subpops)
        self.flaml = flaml
        self.crse_code = '_headcnt'
        self.current = Students(**kwargs)
        kwargs.pop('date',None)
        self.stable  = Students(**kwargs)
        #apply LAG to term_code
        offset = {1:7, 8:93} #yyyy01->yyyy08 (+7); yyyy08->(yyyy+1)01 (+100-7=+93)
        for _ in range(LAG):
            kwargs['term_code'] += offset[kwargs['term_code']%10]
        self.actual  = Students(**kwargs)


    def get_enrollments(self, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        if not self.is_learner:
            return None
        def fcn():
            def fcn1(agg):
                grp = union('crse_code', self.subpops, agg, 'term_desc')
                g = lambda X, Y: get_incoming(X.join(Y, rsuffix='_drop')).groupmy(grp)['count'].sum()  # get stuff from Y that is not in X
                df = pd.DataFrame({
                    'current':g(self.current.get_students(), self.actual.get_registrations()),  # join actual course registrations onto current student info
                    'actual' :g(self.actual.get_registrations(), self.stable.get_students()),  # join stable student info onto actual course registrations
                    }).fillna(0)
                df['mlt'] = df['actual'] / df['current']
                return df.sort_index()
            return {agg: fcn1(agg) for agg in self.aggregates}
        return self.run(fcn, f'{nm}/{self.date}/{self.term_code}', [self.current.get_students, self.stable.get_students, self.actual.get_registrations], suffix='.pkl', **kwargs)[0]


    def get_imputed(self, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            def fcn1(df):
                X = df[self.features.keys()].prep(category=True)
                imp = mf.ImputationKernel(X.reset_index(drop=True), random_state=self.seed)
                imp.mice(10)
                return imp.complete_data().set_index(X.index)
            return {s: fcn1(df) for s, df in self.current.get_students().fillna(self.features).groupmy(self.subpops)}
            # Z = self.current.get_students().fillna(self.features)
            # Z = Z.query('_tot_sch > 0') if LAG > 0 else Z
            # return {s: fcn1(df) for s, df in Z.groupmy(self.subpops)}
        return self.run(fcn, f'{nm}/{self.date}/{self.term_code}', self.current.get_students, suffix='.pkl', **kwargs)[0]


    def get_prepared(self, **kwargs):
        def fcn(X):
            Z = (X
                .join(self.current.reg(self.crse_code).rename('current'))
                .join(self.actual .reg(self.crse_code).rename('actual'))
                .fillna({'current':False, 'actual':False})
            )
            if self.crse_code == '_proba':
                Z = Z.drop(columns=union(races, 'gender', 'international'), errors='ignore')
            return [Z, Z.pop('actual')]
        return {s: fcn(X) for s, X in self.get_imputed().items()}


    def get_learners(self, **kwargs):
        """train model - biggest bottleneck - can we run multiple (crse_code, year) in parallel?"""
        nm = sys._getframe().f_code.co_name[4:]
        if not self.is_learner:
            return None
        def fcn():
            def fcn1(s, Z):
                dct = {
                    'time_budget':30,
                    # 'max_iter': 100,
                    'task':'classification',
                    # 'log_file_name': self.get_dst(f'learners/{self.date}/{self.term_code}/{crse_code}/{s}', suffix='.log')[1],
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
        lrn = self.run(fcn, f'{nm}/{self.date}/{self.term_code}/{self.crse_code}', [self.get_imputed, self.current.get_registrations, self.actual.get_registrations], suffix='.pkl', **kwargs)[0]
        del self[nm]
        return lrn


    def get_predictions(self, modl=None, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        modl = modl or self
        modl.crse_code = self.crse_code
        def fcn():
            data = self.get_prepared()
            learners = modl.get_learners()
            dct = {s: pd.DataFrame({'prediction': l.predict_proba(data[s][0]).T[1], 'actual':data[s][1], 'cv_score': 100*l.best_loss}) for s, l in learners.items()}
            return pd.concat(dct, names=self.subpops).reset_index()
        df = self.run(fcn, f'{nm}/{self.date}/{self.term_code}/{self.crse_code}/{modl.term_code}', [self.get_prepared, self.get_enrollments, modl.get_learners], **kwargs)[0]
        del self[nm]
        return df