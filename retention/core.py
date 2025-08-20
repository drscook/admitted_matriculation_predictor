exec(open('./helpers.py').read())
import codetiming
from pgeocode import Nominatim
from sklearn.metrics.pairwise import haversine_distances
catalog = 'dev.bronze.'
username = globals().get('username', run_qry('select current_user()').squeeze().split('@')[0])
races = [f'race_{r}' for r in ['asian','black','hispanic','native','pacific','white']]
flags_raw = pathlib.Path('/Volumes/aiml/scook/scook_files/admitted_flags_raw')
flags_prc = pathlib.Path('/Volumes/aiml/flags/flags_volume/')
root = pathlib.Path(f'/Volumes/aiml/amp/amp_files/{FORECAST_TERM_CODE}')
const = root/'const'
raw = root/f'{LAG}/raw'
output = root/f'{LAG}/{username}'

class Core(BaseCls):
    def __init__(self, date=now, term_code=FORECAST_TERM_CODE, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(seed)
        self.term_code = int(term_code)
        self.year = self.term_code // 100
        self.term_desc, self.census_date = self.get_terminfo().loc[self.term_code,['term_desc','census_date']]
        self.term_desc = self.term_desc[:-4] + ' ' + self.term_desc[-4:]
        self.date = dt_clip(date, weekday=2).date()
        self.stable_date = dt_clip(self.census_date + pd.to_timedelta(14,'D'), weekday=2).date()
        self.day = (self.stable_date - self.date).days


    def get_dst(self, path, suffix='.parquet', **kwargs):
        L = path.split('/')
        return L[0], (kwargs.get('root', output)/join(L,'/')/join(L,'_')).with_suffix(suffix)


    def run(self, fcn, path, prereq=[], read=True, **kwargs):
        nm, dst = self.get_dst(path, **kwargs)
        if nm in REFRESH and dst not in REFRESHED:
            del self[nm]
            rm(dst)
        if not nm in self:
            if not dst.exists():
                [f() for f in union(prereq)]
                with codetiming.Timer():
                    print(f'creating {dst}', end=': ')
                    self[nm] = dump(dst, fcn())
                    REFRESHED.append(dst)
                    print(f'{dst.stat().st_size/1024**2:.1f}MB', end=': ')
                print(divider)
            else:
                self[nm] = load(dst) if read else None
        return self[nm] if read else self.pop(nm), dst


    def get_terminfo(self, show=False, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            qry = f"""
select
    stvterm_code as term_code
    ,replace(stvterm_desc, ' ', '') as term_desc
    ,stvterm_start_date as start_date
    ,stvterm_end_date as end_date
    ,stvterm_fa_proc_yr as fa_proc_yr
    ,stvterm_housing_start_date as housing_start_date
    ,stvterm_housing_end_date as housing_end_date
    ,sobptrm_census_date as census_date
from
    {catalog}saturnstvterm as A
inner join
    {catalog}saturnsobptrm as B
on
    stvterm_code = sobptrm_term_code
where
    sobptrm_ptrm_code='1'
"""
            df = run_qry(qry, show)
            return df
        return self.run(fcn, nm, root=const, **kwargs)[0].set_index('term_code')


    def get_zips(self, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            df = (
                Nominatim('us')._data  # get all zips
                .prep()
                .rename(columns={'postal_code':'zip'})
                .query("state_code.notnull() & state_code not in [None,'mh']")
            )
            return df
        return self.run(fcn, nm, root=const, **kwargs)[0]


    def get_states(self):
        return union(self.get_zips()['state_code'])


    def get_drivetimes(self, **kwargs):
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            campus_coords = {
                's': '-98.215784,32.216217',
                'm': '-97.432975,32.582436',
                'w': '-97.172176,31.587908',
                'r': '-96.467920,30.642055',
                'l': '-96.983211,32.462267',
                }
            def fcn1():
                url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip"
                gdf = gpd.read_file(url).prep().set_index('zcta5ce20')  # get all ZCTA https://www.census.gov/programs-surveys/geography/guidance/geo-areas/zctas.html
                return gdf.sample_points(size=10, method="uniform").explode().apply(lambda g: f"{g.x},{g.y}")  # sample 10 random points in each ZCT
            M = []
            for k, v in campus_coords.items():
                def fcn2():
                    pts, new = self.run(fcn1, stable, f'drivetimes/pts')
                    L = []
                    i = 0
                    di = 200
                    I = pts.shape[0]
                    while i < I:
                        u = join([v, *pts.iloc[i:i+di]],';')
                        url = f"http://router.project-osrm.org/table/v1/driving/{u}?sources={0}&annotations=duration,distance&fallback_speed=1&fallback_coordinate=snapped"
                        response = requests.get(url).json()
                        L.append(np.squeeze(response['durations'])[1:]/60)
                        i += di
                        print(k,i,round(i/I*100))
                    df = pts.to_frame()[[]]
                    df[k] = np.concatenate(L)
                    return df
                df = self.run(fcn2, f'{nm}/{k}', root=const, **kwargs)[0]
                M.append(df)
            D = pd.concat(M, axis=1).groupmy(level=0).min().stack().reset_index().set_axis(['zip','camp_code','drivetime'], axis=1).prep()
            # There are a few USPS zips without equivalent ZCTA, so we assign them drivetimes for the nearest
            Z = self.get_zips().merge(D.query("camp_code=='s'"), on='zip', how='left').set_index('zip')
            mask = Z['drivetime'].isnull()  # zips without a ZTCA
            Z = np.radians(Z[['latitude','longitude']])
            X = Z[~mask]
            Y = Z[mask]
            M = (
                pd.DataFrame(haversine_distances(X, Y), index=X.index, columns=Y.index) # haversine distance between pairs with and without ZCTA
                .idxmin()  # find nearest ZCTA
                .reset_index()
                .set_axis(['new_zip','zip'], axis=1)
                .merge(D, on='zip', how='left')  # merge the drivetimes for that ZCTA
                .drop(columns='zip')
                .rename(columns={'new_zip':'zip'})
            )
            df = pd.concat([D,M], ignore_index=True)
            return df
        return self.run(fcn, nm, self.get_zips, root=const, **kwargs)[0]


    def get_spriden(self, show=False):
        # Get id-pidm crosswalk so we can replace id by pidm in flags below
        # GA's should not have permissions to run this because it can see pii
        if 'spriden' not in self:
            qry = f"""
            select distinct
                spriden_id as id
                ,spriden_pidm as pidm
                ,spriden_last_name as last_name
                ,spriden_first_name as first_name

            from
                {catalog}saturnspriden as A
            where
                spriden_change_ind is null
                and spriden_activity_date between '2000-09-01' and '2025-09-01'
                and spriden_id REGEXP '^[0-9]+'
            """
            self.spriden = run_qry(qry, show)
        return self.spriden


    def get_flagsday(self, early_stop=10, **kwargs):
        # GA's should not have permissions to run this because it sees pii
        nm = sys._getframe().f_code.co_name[4:]
        counter = 0
        divide = False
        for src in union(flags_raw.iterdir(), reverse=True):
            counter += 1
            if counter > early_stop:
                break
            stem, suff = src.name.lower().split('.')
            if 'melt' in stem or 'admitted' not in stem or suff != 'xlsx':
                print(stem, 'SKIP')
                continue
            # Handles 2 naming conventions that were used at different times
            try:
                current_date = pd.to_datetime(stem[:10].replace('_','-')).date()
                multi = True
            except:
                try:
                    current_date = pd.to_datetime(stem[-6:]).date()
                    multi = False
                except:
                    print(stem, 'FAIL')
                    continue
            book = pd.ExcelFile(src, engine='openpyxl')
            # Again, handles the 2 different versions with different sheet names
            if multi:
                sheets = {int(sheet): sheet for sheet in book.sheet_names if sheet.isnumeric()}
            else:
                sheets = {int(stem[:6]): book.sheet_names[0]}
            for term_code, sheet in sheets.items():
                # if term_code in self.get_terminfo().index:  # not sure whether this is still necessary after recent revision
                def fcn():
                    B = book.parse(sheet).prep()
                    B['id'] = B['id'].prep(errors='coerce')  # CRITICAL step - id is stored as string dtype to allow leading 0's, but this opens the door for serious data entry errors (ex: ID="D") which can have catastrophic effects downstream.  This step convert such issues to null, which get removed during the merge below.
                    mask = B['id'].isnull()
                    if mask.any():
                        print(f'WARNING: {mask.sum()} non-numeric ids')
                        B[mask].disp(5)
                    df = (
                        self.get_spriden()[['pidm','id']]
                        .assign(current_date=current_date)
                        .merge(B, on='id', how='inner')
                        .drop(columns=['id','last_name','first_name','mi','pref_fname','street1','street2','primary_phone','call_em_all','email'], errors='ignore')
                    )
                    df.loc[~df['state'].isin(self.get_states()),'zip'] = pd.NA
                    df['zip'] = df['zip'].str.split('-').str[0].str[:5].prep(errors='coerce')
                    with no_warn(UserWarning):
                        for k in ['dob',*df.filter(like='date').columns]:  # convert date columns
                            if k in df:
                                df[k] = pd.to_datetime(df[k], errors='coerce')
                    return df
                if self.run(fcn, f'{nm}/{current_date}/{term_code}', root=flags_prc, read=False, **kwargs)[1] in REFRESHED:
                    counter = 0
                    rm(self.get_dst(f'flagsyear/{term_code//100}', root=flags_prc)[1])
                del self[nm]


    def get_flagsyear(self, **kwargs):
        # may hit memory limits running get_flagsyear - just restart kernel and try again
        nm = sys._getframe().f_code.co_name[4:]
        def fcn():
            with no_warn(FutureWarning):
                L = [load(src) for src in (flags_prc/'flagsday').rglob(f'**/{self.year}*/*.parquet')]
                df = pd.concat(L, ignore_index=True)
                for X in L:
                    del X
                return df
                # return pd.concat([load(src) for src in (flags_prc/'flagsday').rglob(f'**/{self.year}*/*.parquet')], ignore_index=True)
        return self.run(fcn, f'{nm}/{self.year}', root=flags_prc, **kwargs)[0]