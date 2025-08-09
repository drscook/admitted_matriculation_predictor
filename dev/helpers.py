import  os, sys, pathlib, shutil, pickle, warnings, contextlib, requests, functools, time, dataclasses
import numpy as np, pandas as pd, geopandas as gpd, pyarrow.parquet as pq, matplotlib.pyplot as plt
from IPython.display import clear_output, display
from copy import deepcopy as copy
pd.options.display.max_columns = None
pd.options.future.no_silent_downcasting = True
now = pd.Timestamp.now(tz='America/Chicago').tz_localize(None)
divider = '#'*100

############ Misc functions ############

@contextlib.contextmanager
def no_warn(*categories):
    """contextmanager to suppress warnings"""
    with warnings.catch_warnings():
        for category in categories if categories else [Warning]:
            warnings.filterwarnings("ignore", category=category)
        yield

def safe(fcn=None):
    """if fcn fails, fallback to args[0] else None"""
    @functools.wraps(fcn)
    def wrapper(*args, **kwargs):
        try:
            return fcn(*args, **kwargs)
        except:
            return args[0] if args else None
    return wrapper
    
def setmeth(cls, fcn):
    """monkey-patch new method into a mutable class (fails for immutable class)"""
    setattr(cls, fcn.__name__, fcn)

class BaseCls():
    """Allows self['attr'] and self.attr syntax"""
    def __init__(self, **kwargs):
        pass
    def __contains__(self, key):
        return hasattr(self, key)
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, val):
        setattr(self, key, val)
    def __delitem__(self, key):
        if key in self:
            delattr(self, key)
    def get(self, key, default=None):
        return self[key] if key in self or default is None else default
    def pop(self, key, default=None):
        val = self.get(key, default)
        del self[key]
        return val
    #chainable
    def set(self, key, val):
        self[key] = val
        return self
    def drop(self, key, default=None):
        self.pop(key, default)
        return self

############ List functions ############

def listify(*args, reverse=None):
    """ensure it is a list"""
    if len(args)==1:
        if args[0] is None or args[0] is np.nan or args[0] is pd.NA:
            return list()
        elif isinstance(args[0], str):
            return [args[0]]
    try:
        L = list(*args)
    except Exception as e:
        L = list(args)
    try:
        L = sorted(L, reverse=reverse) 
    except Exception as e:
        pass
    return L

def setify(*args):
    """ensure it is a set"""
    return set(listify(*args))

def unpack(*args, **kwargs):
    L = [y for x in args for y in (unpack(*x) if isinstance(x, (list,tuple,set)) else listify(x))]
    return listify(L, **kwargs)

def union(*args, **kwargs):
    return listify(dict.fromkeys(unpack(*args)), **kwargs)

def intersection(*args, **kwargs):
    L = [union(x, **kwargs) for x in args]
    A = L.pop(0)
    while L:
        B = L.pop(0)
        A = [x for x in A if x in B]
    return A

def difference(A, *args, **kwargs):
    return [x for x in union(A, **kwargs) if x not in union(*args, **kwargs)]

############ String functions ############

def rjust(x, width, fillchar=' '):
    return str(x).rjust(width,str(fillchar))

def ljust(x, width, fillchar=' '):
    return str(x).ljust(width,str(fillchar))

def join(lst, sep='\n,', pre='', post=''):
    """flexible way to join list of strings into a single string"""
    return f"{pre}{str(sep).join(map(str,listify(lst)))}{post}"

############ Pandas functions ############

def setmeth_pd(fcn):
    """monkey-patch helpers into Pandas Series & DataFrame"""
    setmeth(pd.DataFrame, fcn)
    setmeth(pd.Series, fcn)

def wrap1(fcn, X):
    """Make 1-dim fcn work for Series and DataFrames"""
    return X.apply(fcn) if isinstance(X, pd.DataFrame) else fcn(X)

def wrap2(fcn, X):
    """Make 2-dim fcn work for Series and DataFrames"""
    return fcn(pd.DataFrame(X)).squeeze() if isinstance(X, pd.Series) else fcn(X)

def disp(X, n_rows=4, ascending=None):
    """convenient display method"""
    n_rows = n_rows if n_rows > 0 else 1000
    def fcn(df):
        if ascending is not None:
            df = df.rename(columns=lambda x: str(x)).sort_index(axis=1, ascending=ascending)
        miss = df.isnull()
        info = pd.DataFrame({'dtype':df.dtypes.astype('string'), 'missing_cnt':miss.sum(), 'missing_pct':miss.mean().round(3)*100}).T
        print(df.shape)
        display(info)
        with pd.option_context('display.max_rows', n_rows, 'display.min_rows', n_rows):
            display(df)
        return df
    wrap2(fcn, X)
setmeth_pd(disp)

def sort(X, by=None, **kwargs):
    """my preferred defaults for sort_values"""
    kwargs['by'] = by or list(X.columns)
    def fcn(df):
        return df.sort_values(**kwargs)
    return wrap2(fcn, X)
setmeth_pd(sort)

def groupmy(X, by=None, **kwargs):
    """my preferred defaults for groupby"""
    kwargs = {
        'level':None,
        'as_index':True,
        'sort':False,
        'group_keys':False,
        'observed':False,
        'dropna':False,
    } | kwargs
    def fcn(df):
        return df.groupby(by, **kwargs)
    return wrap2(fcn, X)
setmeth_pd(groupmy)

def prep_data(X, str_fcn=None, category=False, downcast='integer', **kwargs):
    """convert to numeric dtypes if possible"""
    import pandas.api.types as tp
    def fcn(ser):
        with no_warn():
            if tp.is_datetime64_any_dtype(ser) or str(ser.dtype) in ['category','geometry']:
                pass
            elif tp.is_string_dtype(ser) or tp.is_object_dtype(ser):
            
                ser = ser.astype('string').apply(safe(str_fcn))
                try:
                    ser = pd.to_numeric(ser, downcast=downcast, **kwargs)
                except ValueError:
                    try:
                        ser = pd.to_datetime(ser)
                    except ValueError:
                        if category:
                            ser = ser.astype('category')
            else:
                ser = pd.to_numeric(ser, downcast=downcast, **kwargs)
            return ser.astype('Int64') if tp.is_integer_dtype(ser) else ser.convert_dtypes()
    return wrap1(fcn, X)
setmeth_pd(prep_data)

def prep(X, str_fcn=lambda x: x.strip().lower(), col_fcn=lambda x: x.strip().lower().replace(' ','_').replace('-','_'), **kwargs):
    def fcn(df):
        L = [prep_data(Y, str_fcn, **kwargs).rename(columns=safe(col_fcn)) for Y in [df.reset_index(drop=True), df[[]].reset_index()]]
        return L[0].set_index(pd.MultiIndex.from_frame(L[1]))
    return wrap2(fcn, X)
setmeth_pd(prep)

def prepr(X, **kwargs):
    if isinstance(X, (pd.DataFrame,pd.Series)):
        return X.prep(**kwargs)
    elif isinstance(X, dict):
        return {k: prepr(v, **kwargs) for k, v in X.items()}
    elif isinstance(X, (list,tuple,set)):
        return type(X)(prepr(v, **kwargs) for v in X)
    else:
        return X
    
def dt_clip(X=now, weekday=None, lower=None, upper=None):
    def fcn(ser):
        s = pd.Series(pd.to_datetime(ser))
        l = pd.to_datetime(lower)
        u = pd.to_datetime(upper)
        if l and weekday and l.weekday() > weekday:
            l += pd.to_timedelta(7, 'D')
        if u and weekday and u.weekday() < weekday:
            u -= pd.to_timedelta(7, 'D')
        s = s.clip(lower=l, upper=u)
        if weekday:
            s += pd.to_timedelta(weekday - s.dt.weekday, 'D')
        return s.squeeze()
    return wrap1(fcn, X)
setmeth_pd(dt_clip)

############ SQL functions ############

def indent(x, lev=1):
    return x.replace('\n','\n'+'    '*lev) if lev>0 else x

def subqry(qry, lev=1):
    """make qry into subquery"""
    qry = '\n' + qry.strip()
    qry = '(' + qry + '\n)' if 'select' in qry else qry
    return indent(qry, lev)

def coalesce(x, y=False):
    return f'coalesce({x}, {y}) as {x}'

def run_qry(qry, show=False, sample='10 rows', seed=42):
    """run qry and return dataframe"""
    L = qry.split(' ')
    if len(L) == 1:
        qry = f'select * from {L[0]}'
        if sample is not None:
            qry += f' tablesample ({sample}) repeatable ({seed})'
    if show:
        print(qry)
    return pd.DataFrame(spark.sql(qry).toPandas()).prep()

############ File I/O functions ############

def get_size(path):
    return os.system(f'du -sh {path}')

def pathify(path):
    return pathlib.Path(path)

def mkdir(path):
    p = pathify(path)
    (p if p.suffix == '' else p.parent).mkdir(parents=True, exist_ok=True)
    return p

def rm(path, root=False):
    p = pathify(path)
    if p.is_file():
        p.unlink()
    elif p.is_dir():
        if root:
            shutil.rmtree(p)
        else:
            for q in p.iterdir():
                rm(q, True)
    return p

def reset(path):
    return mkdir(rm(path))

def dump(path, obj, **kwargs):
    p = reset(path)
    obj = prepr(obj, **kwargs)
    if p.suffix == '.parquet':
        obj.to_parquet(p, **kwargs)
    elif p.suffix in ['.csv','.txt']:
        obj.to_csv(p, **kwargs)
    else:
        with open(p, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL, **kwargs)
    return obj

def load(path, **kwargs):
    p = pathify(path)
    if p.suffix == '.parquet':
        try:
            obj = gpd.read_parquet(p, **kwargs)
        except:
            obj = pd.read_parquet(p, **kwargs)
    elif p.suffix in ['.csv','.txt']:
        obj = pd.read_csv(p, **kwargs)
    else:
        try:
            obj = gpd.read_file(path, **kwargs)
        except:
            with open(p, 'rb') as f:
                obj = pickle.load(f, **kwargs)
    return prepr(obj)