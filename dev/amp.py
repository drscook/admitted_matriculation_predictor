exec(open('./term.py').read())
class AMP(Core):
    def __init__(self, start=2023, crse_codes=None, **kwargs):
        kwargs['term_code'] = FORECAST_TERM_CODE
        super().__init__(**kwargs)
        self.crse_codes = union('_headcnt', '_proba', crse_codes)
        kwargs.pop('date')
        kwargs.pop('term_code')
        self.terms = {t: Term(
            date=self.date-pd.Timedelta(days=365*k),
            term_code=t,
            is_learner=t<self.term_code,
            **kwargs) for k, t in enumerate(range(self.term_code, 100*start, -100))}


    def get_predictions(self):
        L = [pred_obj.set('crse_code', crse_code).get_predictions(model_obj).assign(
                crse_code=crse_code,
                prediction_term_code=pred_code,
                model_term_code=model_code,
                )
        for crse_code in self.crse_codes for pred_code, pred_obj in self.terms.items() for model_code, model_obj in self.terms.items() if model_obj.is_learner]
        return pd.concat(L)