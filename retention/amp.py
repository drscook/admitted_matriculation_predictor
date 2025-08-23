exec(open('./term.py').read())
class AMP(Core):
    def __init__(self, start=2022, crse_codes=['_headcnt','_proba'], **kwargs):
        kwargs['term_code'] = FORECAST_TERM_CODE
        super().__init__(**kwargs)
        self.crse_codes = union(crse_codes)
        kwargs.pop('date',None)
        kwargs.pop('term_code',None)
        self.terms = {t: Term(term_code=t, is_learner=t<self.term_code, date=self.date-pd.Timedelta(days=365*k), **kwargs) for k, t in enumerate(range(self.term_code, 100*start, -100))}
        self.srt = lambda df: df.sort_values(list(df.columns), ascending=['term' not in k for k in df.columns])
        for k in ['subpops', 'aggregates']:
            self[k] = self.terms[self.term_code][k]


    def setup(self):
        for fcn in [t['get_'+k] for k in ['flagsyear','imputed'] for t in self.terms.values()]:
            fcn(read=False)
        self.get_enrollmentsall()
        self.subqry = f"""from parquet.`{self.get_predictionsall(read=False)[1]}` as A left join parquet.`{self.get_studentsall(read=False)[1]}` as B using ({join(union('pidm', self.subpops, 'cohort_term'))})"""
        return self


    def get_studentsall(self, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            print()
            with no_warn():
                return pd.concat([t.current.get_students() for t in self.terms.values()]).rename(columns={'term_desc':'cohort_term'}).reset_index()
        return self.run(fcn, f'{nm}/{self.date}', **kwargs)


    def get_predictionsall(self, **kwargs):
            nm = sys._getframe().f_code.co_name[4:]
            def fcn():
                print()
                Z = pd.concat([pred.set('crse_code', crse_code).get_predictions(modl).assign(crse_code=crse_code, prediction_term=pred.actual.term_desc, cohort_term=pred.current.term_desc, model_term=modl.current.term_desc) for crse_code in self.crse_codes for pred in self.terms.values() for modl in self.terms.values() if modl.is_learner]).reset_index(drop=True)
                E = self.get_enrollmentsall()['crse_code'].loc['_headcnt']['mlt'].rename_axis(index={'term_desc':'model_term'}).reset_index()
                return Z.merge(E, how='left')
            return self.run(fcn, f'{nm}/{self.date}', self.get_enrollmentsall, **kwargs)


    def get_proba(self, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            df = run_qry(f"select * except (crse_code, cv_score, mlt) {self.subqry} where crse_code='_proba'")
            return self.srt(df.move('prediction_term', 'cohort_term').move('model_term', 'cohort_term', 1))
        return self.run(fcn, f'{nm}/{self.date}/{self.term_code}', self.setup, **kwargs)[0]
    
    
    def get_enrollmentsall(self, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            print()
            return {k: pd.concat([t.get_enrollments()[k] for t in self.terms.values() if t.is_learner]) for k in self.aggregates}
        return self.run(fcn, f'{nm}/{self.date}', suffix='.pkl', **kwargs)[0]


    def get_forecasts(self, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            def fcn1(agg):
                group = join(union('crse_code', self.subpops, agg, 'prediction_term', 'cohort_term', 'model_term'))
                qry = f"select {group}, sum(prediction*mlt) as prediction, max(cv_score) as cv_score {self.subqry} where crse_code<>'_proba' group by {group}"
                df = run_qry(qry).merge(self.get_enrollmentsall()[agg]['actual'].rename_axis(index={'term_desc':'prediction_term'}).reset_index(), how='left').fillna(0)
                df['error'] = df['prediction'] - df['actual']
                df['error_pct'] = df['error'] / df['actual'].replace(0, pd.NA) * 100
                df['cv_score'] = df.pop('cv_score')
                df.loc[df.eval('prediction_term == @self.term_desc'), ['actual', 'error', 'error_pct']] = pd.NA
                return self.srt(df)
            return {agg: fcn1(agg) for agg in self.aggregates}
        return self.run(fcn, f'{nm}/{self.date}/{self.term_code}', self.setup,  suffix='.pkl', **kwargs)[0]


    def get_reports(self):
        rpts = {
            'summary':lambda df: df.query(f"prediction_term==prediction_term.max() & model_term==model_term.max()").loc[:,:'prediction'],
            'details':lambda df: df,
        }
        rpts = {rpt: {agg: fcn(df).round().prep() for agg, df in self.get_forecasts().items()} for rpt, fcn in rpts.items()}
        rpts['students'] = {join(s): df for s, df in self.get_spriden().merge(self.get_proba().query(f"prediction_term==prediction_term.max() & model_term==model_term.max()"), how='right').groupmy(self.subpops)}

        instructions = pd.DataFrame({f"Admitted Matriculation Projections (AMP) for {self.date}":[
            f'''Executive Summary''',
            f'''AMP is a predictive model designed to forecast the incoming (not continuing) Fall cohort to help leaders proactively allocate resources (instructors, sections, labs, etc) in advance.''',
            f'''It is intended to supplement, not replace, the knowledge, expertise, and insights developed by institutional leaders through years of experience.''',
            f'''Like all AI/ML models (and humans), AMP is fallible and should be used with caution.''',
            f'''AMP learns exclusively from historical data captured in EM’s Flags reports and IDA’s daily admissions and registration snapshots.''',
            f'''It cannot account for factors not present in these datasets, including curriculum changes, policy shifts, structural changes, demographic variation, changes in oversight, etc.''',
            f'''''',
            f'''AMP provides both “Summary” and “Details” files. For most users, rows in the “Summary” file with model_term_code = 202408 will suffice.''',
            f'''Because AMP’s accuracy varies across courses, the “Details” file includes historical error analyses to help users assess the reliability of each forecast (details below).''',
            '',
            f'''As widely requested, AMP includes predictions for the Fall 2025 cohort in Ft. Worth, despite having no prior Ft. Worth FTIC example to learn from.''',
            f'''These are a good-faith effort to offer my best data-driven insights, but due to the lack of training data,''',
            f'''they are inherently more speculative and should be treated with lower confidence (details below).''',
            f'''''',
            f'''Definitions''',
            f'''crse_code = course code (_headcnt = total headcount)''',
            f'''styp_desc = student type; returning = re-enrolling after a previous attempt (not continuing)''',
            f'''prediction_term = term being forecast''',
            f'''cohort_term = cohort being forecast''',
            f'''model_term = cohort used to train AMP''',
            f'''prediction = forecast headcount''',
            f'''*actual = true headcount''',
            f'''*error = prediction - actual''',
            f'''*error_pct = error / actual * 100''',
            f'''*cv_score = average validation log-loss from 3-fold cross-validation''',
            f'''*=appears only in “Details” & not available for 2025 (since actuals are not yet known)''',
            '',
            f'''Methodology''',
            f'''AMP uses XGBoost, a machine learning algorithm, to forecast the number, characteristics, and likely course enrollments of incoming Fall students.''',
            f'''Predictions are based on application and pre-semester engagement (orientation, course registration, financial aid, etc.) from EM’s Flags and IDA’s daily snapshots.''',
            f'''For each student admitted for Fall 2025, AMP identifies similar students from past Fall cohorts, analyzes their course enrollments (if any),''',
            f'''learns relevant patterns, then forecasts Fall 2025 course enrollment for the admitted student in question.''',
            f'''More precisely, for each (incoming student, course)-pair, AMP assigns a probability whether that student will be enrolled in that course on the Friday after Census.''',
            f'''These (student, course)-level probabilities are then aggregated in many different ways to forecast headcounts for courses, campuss, majors, colleges, TSI statuses, etc.''',
            f'''These appear on different sheets in this workbook.''',
            f'''''',
            f'''Since admissions and registration data evolve through the spring and summer, AMP is trained only on data available as of the same date in previous years.''',
            f'''AMP's forecast for Ft. Worth's Fall 2025 cohort are necessarily based on previous Stephenville cohorts since no Ft. Worth FTIC's existed on this date.''',
            f'''Suppose AMP predicts, "Based on similar FTIC's in Stephenville in 2024, I predict Alice has a 75% probability to matriculate in Fall 2025".''',
            f'''If Alice is applying to Ft. Worth, then 0.75 is added to Ft. Worth's forecast.''',
            f'''However, AMP can not yet understand how to adjust its 75% projection to reflect how Ft. Worth FTIC's behave differently than Stephenville FTIC since there are no Ft. Worth FTIC's to learn from.''',
            f'''Though not ideal, this is the best idea we've found to forecast Ft. Worth FTIC in the absence of valid training examples.''',
            f'',
            f'''AMP is trained separately using the Fall 2024, 2023, and 2022 cohorts.''',
            f'''Most users should focus on prediction_term_code = 202508 and model_term_code = 202408, as Fall 2025 is likely to resemble Fall 2024 more closely than Fall 2023 & Fall 2022.''',
            f'''Users with domain expertise may choose to incorporate older cohorts (e.g., weighted average of model_term_codes 2024, 2023, & 2022) if they believe those terms are similarly relevant.''',
            f'',
            f'''Rows for prediction_term_code < 202508 appear only in the “Details” file and include retrospective "predictions" and actual outcomes.''',
            f'''This allows users to assess AMP's ability to forecast each individual course and calibrate their confidence accordingly.''',
            f'',
            f'''Predictions for small values are less reliable than for large numbers (Central Limit Theorem).''',
            f'',
            f'''AMP only models students who have already applied and been admitted (eager).''',
            f'''However, more students will apply between now and start of term, especially transfer & returning (lagging).''',
            f'''AMP generates forecasts based on eager students then inflates using the eager-lagging ratio from that model_term_code.''',
            f'''This assumes the eager-lagging behavior will be approximately the same this year.''',
            f'''While this assumption cannot be verified in advance, we must make SOME assumption. This one has proven sufficiently accurate in past cycles.''',
            f'',
            f'''Dr. Scott Cook is eager to provide as much additional detail as the user desires: scook@tarleton.edu.''',
            f'''source code: https://github.com/drscook/admitted_matriculation_predictor'''
        ]})

        def format_xlsx(sheet):
            from openpyxl.styles import Alignment
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.alignment = Alignment(horizontal="left")
            for column in sheet.columns:
                width = 1+max(len(str(cell.value))+3*(i==0) for i, cell in enumerate(column))
                sheet.column_dimensions[column[0].column_letter].width = width
            sheet.freeze_panes = "A2"

        def write_xlsx(rpt, sheets):
            nm, dst = self.get_dst(f'reports/{self.date}/{rpt}', suffix='.xlsx')
            reset(dst)
            print(f'creating {dst}', end=': ')
            with codetiming.Timer():
                src = "./report.xlsx"
                sheets = {'instructions':instructions} | sheets
                with pd.ExcelWriter(src, mode="w", engine="openpyxl") as writer:
                    for sheet_name, df in sheets.items():

                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        format_xlsx(writer.sheets[sheet_name])
                shutil.copy(src, dst)
                rm(src)
            print(divider)
            return sheets

        [write_xlsx(rpt, sheets) for rpt, sheets in rpts.items()]
        return rpts