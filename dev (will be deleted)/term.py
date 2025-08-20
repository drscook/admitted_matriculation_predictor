exec(open('./students.py').read())
import miceforest as mf
with no_warn():
    import flaml as fl
from sklearn.metrics import log_loss

def custom_log_loss(X_val, y_val, estimator, labels, X_train, y_train, *args, **kwargs):
    """Flaml's log_loss errs when only 1 value appears in a split, so we create custom_log_loss to specify labels: https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/"""
    #I think there is an easier solution, but I can't find it.
    val_loss = log_loss(y_val, estimator.predict_proba(X_val), labels=[False,True])
    train_loss = log_loss(y_train, estimator.predict_proba(X_train), labels=[False,True])
    return val_loss, {"val_loss": val_loss, "train_loss": train_loss}

class Term(Core):
    def __init__(self, is_learner=True, features=dict(), subpops='styp_desc', aggregates=None, flaml=dict(), **kwargs):
        super().__init__(**kwargs)
        self.is_learner = is_learner
        self.features = features
        self.subpops = union(subpops)
        self.aggregates = difference(union('crse_code', aggregates), subpops)
        self.flaml = flaml
        self.current = Students(**kwargs)
        kwargs['date'] = self.stable_date
        self.stable = Students(**kwargs)
        self.crse_code = '_headcnt'


    def get_enrollments(self, **kwargs):
        def fcn():
            def fcn1(agg):
                grp = union('crse_code', self.subpops, agg)
                g = lambda X, Y: get_incoming(X).join(Y, rsuffix='_drop').groupmy(grp)['count'].sum()  # get stuff from Y that is not in X
                df = pd.DataFrame({
                    'current':g(self.current.get_students(), self.stable.get_registrations()),  # join stable course registrations onto current list of students
                    'stable' :g(self.stable.get_registrations(), self.stable.get_students()),  # join extra student info onto stable list of registrations
                    }).fillna(0)
                df['mlt'] = df['stable'] / df['current']
                return df.sort_index()
            return {agg: fcn1(agg) for agg in self.aggregates}
        return self.run(fcn, f'enrollments/{self.date}/{self.term_code}', [self.current.get_students, self.stable.get_students, self.stable.get_registrations], suffix='.pkl', **kwargs)[0]


    def get_imputed(self, **kwargs):
        def fcn():
            def fcn1(df):
                X = df.fillna(self.features)[self.features.keys()].prep(category=True)
                imp = mf.ImputationKernel(X.reset_index(drop=True), random_state=self.seed)
                imp.mice(10)
                return imp.complete_data().set_index(X.index)
            return {s: fcn1(df) for s, df in self.current.get_students().groupmy(self.subpops)}
        return self.run(fcn, f'imputed/{self.date}/{self.term_code}', self.current.get_students, suffix='.pkl', **kwargs)[0]


    def get_prepared(self):
        def fcn(X):
            g = lambda k, v=self.crse_code: self[k].get_registrations().query(f"crse_code==@v")['count'].rename(k if v==self.crse_code else v)
            Z = (X
                .join(g('current', '_tot_sch'))
                .join(g('current').astype('boolean'))
                .join(g('stable' ).astype('boolean'))
                .fillna({'_tot_sch':0, 'current':False, 'stable':False})
            )
            if self.crse_code == '_proba':
                Z = Z.drop(columns=union(races, 'gender', 'international'), errors='ignore')
            return [Z, Z.pop('stable')]
        return {s: fcn(X) for s, X in self.get_imputed().items()}


    def get_learners(self, **kwargs):
        """train model - biggest bottleneck - can we run multiple (crse_code, year) in parallel?"""
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
        return self.run(fcn, f'learners/{self.date}/{self.term_code}/{self.crse_code}', [self.get_imputed, self.current.get_registrations, self.stable.get_registrations], suffix='.pkl', **kwargs)[0]


    def get_predictions(self, modl=None, **kwargs):
        modl = modl or self
        modl.crse_code = self.crse_code
        def fcn():
            data = self.get_prepared()
            learners = modl.get_learners()
            dct = {s: pd.DataFrame({'prediction': l.predict_proba(data[s][0]).T[1], 'actual':data[s][1], 'cv_score': 100*l.best_loss}) for s, l in learners.items()}
            return pd.concat(dct, names=self.subpops)
        return self.run(fcn, f'predictions/{self.date}/{self.term_code}/{self.crse_code}/{modl.term_code}', [self.get_prepared, modl.get_learners], **kwargs)[0]