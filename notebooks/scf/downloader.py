import re
from urllib import request

pattern = re.compile(r' +')

try:
    url = "http://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt"

    req = request.urlopen(url)

    raw_data = req.read().decode('utf-8').strip('\n')
except:
    req = open('./data/data.txt')

    raw_data = req.read().strip('\n')

raw_data = pattern.sub(' ', raw_data)
raw_data = raw_data.split('\n')[:-1]
raw_data = map(lambda l: l.split(' '), raw_data)
raw_data = list(raw_data)
