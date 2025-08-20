exec(open('./core.py').read())
def get_desc(code):
    for nm in code.split('_'):
        if len(nm) == 4:
            break
    return f'{code} as {nm}_code, (select stv{nm}_desc from {catalog}saturnstv{nm} where {code} = stv{nm}_code limit 1) as {nm}_desc'

def coalesce(x, y=False):
    return f'coalesce({x}, {y}) as {x}'

def get_incoming(df):
    return df.query("levl_code=='ug' & styp_code in ['n','r','t']")

def get_duplicates(df, subset='pidm', n_rows=10, errors='raise'):
    ct = df.groupmy(subset).transform('size')
    if ct.max() > 1:
        df.insert(0, 'ct', ct)
        df.query('ct>1').drop_duplicate(subset=subset).sort_values('ct', ascending=False).head(n_rows).disp(n_rows)
        if errors == 'raise':
            raise Exception('duplicates detected')

class Students(Core):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def get_flags(self, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            df = self.get_flagsyear().query(f"current_date<=@self.date").sort_values(['current_date','app_date']).drop_duplicates(subset='pidm', keep='last')
            # df = self.get_flagsyear().query(f"current_date<=@self.date").sort_values('current_date').drop_duplicates(subset=['pidm','term_code'], keep='last')
            del self['flagsyear']
            return df
        return self.run(fcn, f'{nm}/{self.date}/{self.term_code}', self.get_flagsyear, root=raw, **kwargs)[0]


    def newest(self, qry, prt, tbl='', sel=''):
        """The OPEIR daily snapshot experienced occasional glitched causing incomplete copies.
        Consequently, records can have a "gap" where they vanish then reappear later. This function fixes this issue."""
        prt = join(prt, ', ')
        if tbl == '':
            tbl = qry
        if sel != '':
            sel = ','+join(sel, '\n,')

        qry = f"""
select
    {prt}
    ,current_date
    ,min(current_date) over (partition by {prt}) as first_date --first date this record appeared
    ,max(current_date) over (partition by {prt}) as last_date  --last  date this record appeared
    ,least(timestamp('{self.date}'), max(current_date) over ()) as cutoff_date  --cutoff_date is the ealier of self.date and last date of ANY record
from
    {qry.strip()}
qualify
    cutoff_date between first_date and dateadd(last_date, 5)  -- keep records where cutoff_date falls between that record's first & last appearance (+5 days margin handles rare technical glitch where a record for this upcoming term is missing TODAY but might reappear soon ... does not apply for prior terms).
"""

        qry = f"""
select
    *
from {subqry(qry)}
where
    current_date <= cutoff_date  -- discard records after cutoff_date
qualify
    row_number() over (partition by {prt} order by current_date desc) = 1  -- keep most recent remaining record
"""

        qry = f"""
select distinct
    pidm
    ,current_date
    ,first_date
    ,last_date
    ,{get_desc('term_code')}
    ,{get_desc('levl_code')}
    ,{get_desc('styp_code')}
    ,{get_desc('camp_code')}
    ,{get_desc('coll_code_1')}
    ,{get_desc('dept_code')}
    ,{get_desc('majr_code_1')}
    --,gender
    ,spbpers_sex as gender
    ,birth_date
    ,{get_desc('spbpers_lgcy_code')}
    ,gorvisa_vtyp_code is not null as international
    ,gorvisa_natn_code_issue as natn_code, (select stvnatn_nation from {catalog}saturnstvnatn where gorvisa_natn_code_issue = stvnatn_nation limit 1) as natn_desc
    ,{coalesce('race_asian')}
    ,{coalesce('race_black')}
    ,coalesce(spbpers_ethn_cde=2, False) as race_hispanic
    ,{coalesce('race_native')}
    ,{coalesce('race_pacific')}
    ,{coalesce('race_white')}
    {indent(sel)}
from {subqry(qry)} as A

left join
    {tbl}
using
    ({prt}, current_date)

left join
    {catalog}spbpers_v
on
    pidm = spbpers_pidm

left join (
    select
        *
    from
        {catalog}generalgorvisa
    qualify
        row_number() over (partition by gorvisa_pidm order by gorvisa_seq_no desc) = 1
    )
on
    pidm = gorvisa_pidm

left join (
    select
        gorprac_pidm
        ,max(gorprac_race_cde='AS') as race_asian
        ,max(gorprac_race_cde='BL') as race_black
        ,max(gorprac_race_cde='IN') as race_native
        ,max(gorprac_race_cde='HA') as race_pacific
        ,max(gorprac_race_cde='WH') as race_white
    from
        {catalog}generalgorprac
    group by
        gorprac_pidm
    )
on
    pidm = gorprac_pidm
"""
        return qry


    def get_registrations(self, show=False, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            tbl = f'dev.opeir.registration_{self.term_desc}'.replace(' ','')
            if spark.catalog.tableExists(tbl):
                qry = self.newest(
                    tbl = tbl,
                    prt = ['pidm','crn'],
                    sel = ['credit_hr as count', 'subj_code || crse_numb as crse_code'],
                    qry = f"""
    {tbl} as A
where
    credit_hr > 0
    and subj_code <> 'INST'""")
                A = run_qry(qry, show).sort()
                B = A.drop(columns=['count','crse_code']).groupmy('pidm').last()  #non-crse student info
                C = A.groupmy(['pidm','crse_code'])[['count']].sum().reset_index('crse_code') #combine if enrolled in >1 crn for same crse_code (rare)
                D = C.groupmy('pidm')[['count']].sum() #compute total sch
                E = D.copy()
                C['count'] = 1 #crse_code headcount
                E['count'] = 1 #overall headcount
                F = pd.concat([C, D.assign(crse_code='_tot_sch'), E.assign(crse_code='_headcnt'), E.assign(crse_code='_proba')])
                df = B.join(F).reset_index()
            else:
                # placeholder if table DNE
                # raise Exception(tbl, 'is missing')
                # df = pd.DataFrame(columns=union('pidm','term_desc','levl_code','styp_code',self.subpops,self.features.keys(),'count','crse_code'))
                df = pd.DataFrame(columns=['pidm','levl_code','styp_code','count','crse_code'])
            get_duplicates(df, ['pidm','crse_code'])
            return df
        return self.run(fcn, f'{nm}/{self.date}/{self.term_code}', root=raw, **kwargs)[0].set_index('pidm')


    def get_admissions(self, show=False, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            def fcn1(season):
                tbl = f'dev.opeir.admissions_{season}{self.year}_v'.replace(' ','')
                return self.newest(
                    tbl = tbl,
                    prt = ['pidm', 'appl_no'],
                    sel = [
                        'appl_no',
                        get_desc('apst_code'),
                        get_desc('apdc_code'),
                        get_desc('admt_code'),
                        get_desc('saradap_resd_code'),
                        'hs_percentile',
                        # 'sbgi_code',
                    ],
                    qry = f"""
    {tbl} as A
inner join
    {catalog}saturnstvapdc as B
on
    apdc_code = stvapdc_code
where
    stvapdc_inst_acc_ind is not null  --only accepted""")
            #some incoming fall students decide to take classes during the preceeding summer and therefore may show up as summer applicants
            #we need to capture them, so we pull both summer & fall admission then uniquify later
            L = [run_qry(fcn1(season), show) for season in ['summer','fall']]
            #some pidms have multiple applications with different levl_code
            #goal: exclude a pidm if it has ANY application where levl_coee!='ug' (even if it also has one with levl_code=='ug'), then keep the highest appl_no for each pidm
            df = (
                pd.concat(L, ignore_index=True)
                .sort_values(['levl_code','appl_no'], ascending=[False,True]) #sorts levl_code='ug' to top and any levl_code!='ug' to bottom
                .drop_duplicates(subset='pidm', keep='last') #for each pidm, keeps a levl_code!='ug' row (if any) otherwise highest levl_code='ug' appl_no ... we will remove non-ug rows below
            )
            return get_incoming(df)
        return self.run(fcn, f'{nm}/{self.date}/{self.term_code}', root=raw, **kwargs)[0]


    def get_students(self, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            df = (
                self.get_admissions()
                # .merge(self.get_flags(), on=['pidm','term_code'], how='left', suffixes=['', '_drop'])
                .merge(self.get_flags(), on='pidm', how='left', suffixes=['', '_drop'])
                .merge(self.get_drivetimes(), on=['zip','camp_code'], how='left', suffixes=['', '_drop'])
                .prep()
            )
            get_duplicates(df)
            M = df.query('current_date_drop.isnull()') #admissions missing flags record
            if not M.empty:
                M.sort_values('first_date').disp(10)
                if M.shape[0] > 10:
                    raise Exception('Too many unmatched incoming students')
            #because we allowed summer & fall admissions thru, we need adjust all to fall
            df['term_code'] = self.term_code
            df['term_desc'] = self.term_desc
            for c in ['gap_score','t_gap_score','ftic_gap_score']:
                if c not in df:
                    df[c] = pd.NA
            df['gap_score'] = np.where(
                df['styp_code']=='n',
                df['ftic_gap_score'].combine_first(df['t_gap_score']).combine_first(df['gap_score']),
                df['t_gap_score'].combine_first(df['ftic_gap_score']).combine_first(df['gap_score']))
            df['oriented'] = df['orientation_hold_exists'].isnull() | df['orien_sess'].notnull() | df['registered'].notnull()
            df['verified'] = df['selected_for_ver'].isnull() | df['ver_complete'].notnull()
            df['sat10_total_score'] = (36-9) / (1600-590) * (df['sat10_total_score']-590) + 9
            df['act_equiv'] = df[['act_new_comp_score','sat10_total_score']].max(axis=1)
            df['eager'] = (pd.to_datetime(self.stable_date) - df['first_date']).dt.days
            df['age'  ] = (pd.to_datetime(self.stable_date) - df['birth_date']).dt.days
            for k in ['reading', 'writing', 'math']:
                df[f'tsi_{k}'] = ~df[k].isin(['not college ready', 'retest required', pd.NA, None, np.nan])
            repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
            df['hs_qrtl'] = pd.cut(df['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(df['apdc_code'].map(repl))
            df['lgcy'] = ~df['lgcy_code'].isin(['o', pd.NA, None, np.nan])
            df['resd'] = df['resd_code'] == 'r'
            for k in ['waiver_desc','fafsa_app','ssb_last_accessed','finaid_accepted','schlship_app']:
                df[k.split('_')[0]] = df[k].notnull()
            return df.drop(columns=df.filter(like='_drop'))
        return self.run(fcn, f'{nm}/{self.date}/{self.term_code}', [self.get_drivetimes, self.get_flags, self.get_admissions], root=raw, **kwargs)[0].set_index('pidm')