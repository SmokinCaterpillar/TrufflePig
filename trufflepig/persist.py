import pandas as pd
import sqlite3
import json
import logging
import gc


logger = logging.getLogger(__name__)


JSON = 'JSON_'
SETJSON = 'SETJSON_'
TIMESTAMP = 'TIMESTAMP_'


def to_sqlite(frame, filename, tablename, index=True):
    """ Stores a data frame to sqlite

    Parameters
    ----------
    frame: DataFrame
    filename: str
    tablename: str
    index: bool

    """
    con = sqlite3.connect(filename)

    store_frame = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        first_element = frame[col].iloc[0]
        if isinstance(first_element, (list, dict, tuple)):
            store_frame[JSON+col] = frame[col].apply(json.dumps)
        elif isinstance(first_element, set):
            store_frame[SETJSON+col] = frame[col].apply(lambda x:
                                                        json.dumps(tuple(x)))
        elif isinstance(first_element, pd.Timestamp):
            store_frame[TIMESTAMP+col] = frame[col]
        else:
            store_frame[col] = frame[col]

    logger.info('Storing as SQLITE file {}'.format(filename))
    store_frame.to_sql(tablename, con, index=index)

    logger.info('Garbage Collecting')
    del store_frame
    gc.collect()


def from_sqlite(filename, tablename=None, query=None,
                index_col='index', indexname=None):
    """ Loads a DataFrame from SQLite file

    Parameters
    ----------
    filename: str
    tablename: str
    query: str or None
    index_col: str
    indexname: str

    Returns
    -------
    DataFrame

    """
    if query is None and tablename is None:
        raise ValueError('Need table or query')
    elif query is not None and tablename is not None:
        raise ValueError('Select either tablename or query')

    con = sqlite3.connect(filename)

    if query is None:
        query = 'SELECT * FROM {}'.format(tablename)

    load_frame = pd.read_sql(query, con, index_col=index_col)

    result_frame = pd.DataFrame(index=load_frame.index)
    result_frame.index.name = indexname

    for col in list(load_frame.columns):
        if col.startswith(JSON):
            name = col[len(JSON):]
            result_frame[name] = load_frame[col].apply(json.loads)
            load_frame.drop(col, axis=1, inplace=True)
        elif col.startswith(SETJSON):
            name = col[len(SETJSON):]
            result_frame[name] = load_frame[col].apply(lambda x:
                                                       set(json.loads(x)))
            load_frame.drop(col, axis=1, inplace=True)
        elif col.startswith(TIMESTAMP):
            name = col[len(TIMESTAMP):]
            result_frame[name] = load_frame[col].apply(pd.to_datetime)
        else:
            result_frame[col] = load_frame[col]

    logger.info('Garbage Collecting')
    del load_frame
    gc.collect()

    return result_frame
