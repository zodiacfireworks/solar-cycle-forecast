from pandas import Timestamp
import re
from urllib import request

from numpy import nanmean
from numpy import nanstd
from numpy import nansum
from numpy import isfinite
from numpy import isnan
from numpy import mean
from numpy import sqrt
from numpy import nan

__all__ = [
    'get_data',
    'parse_data',
    'agg_mean',
    'agg_std',
    'agg_quadsum',
    'agg_all',
    'weighted_average',
]

WHITE_SPACE_PATTERN = re.compile(r' +')


def _is_flag(ch):
    """Checks if character `ch` is a flag character ('*')."""
    return ch == '*'


def _fmt_pairing(x):
    """Generates pairs of data in `x` array and the corresponding format 
       function.
    """
    fmt = [Timestamp, float, float, _is_flag]
    return list(zip(fmt, ['-'.join(x[0:3])] + x[4:6] + [x[-1]]))


def get_data():
    try:
        url = "http://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt"

        req = request.urlopen(url)

        raw_data = req.read().decode('utf-8').strip('\n')
    except:
        req = open('./data/data.txt')

        raw_data = req.read().strip('\n')

    raw_data = WHITE_SPACE_PATTERN.sub(' ', raw_data)
    raw_data = raw_data.split('\n')[:-1]
    raw_data = map(lambda l: l.split(' '), raw_data)
    raw_data = list(raw_data)

    return raw_data


def parse_data(x):
    return list(map(lambda y: y[0](y[1]), _fmt_pairing(x)))


def agg_mean(x):
    if all(isnan(x)):
        return nan

    return nanmean(x)


def agg_std(x):
    if all(isnan(x)):
        return nan

    return nanstd(x)


def agg_quadsum(x):
    if all(isnan(x)):
        return nan

    return sqrt(sum(x[isfinite(x)]**2))


def agg_all(x):
    return all(x)


def weighted_average(w):
    def average(x):
        if (all(isnan(x))):
            return nan

        return nansum(w * x)

    return average
