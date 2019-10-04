
import os
import json
import sys

# os.chdir('../')
path = os.getcwd() 
sys.path.insert(0, path)
sys.path.insert(0, path + '/model/')

json_path = path + '/data/config.json'

from utils import Config
config = Config(json_path)
import model
import pandas as pd

####################################################
####################################################

prep_version = sys.argv[1]
data_name = sys.argv[2]

# prep_version = 'basic'
# data_name = 'train'

####################################################
####################################################

preprocess = getattr(sys.modules['model'].preprocess,prep_version)
data = pd.read_csv(config.dict['origin']['path'][data_name])
prep_data = preprocess(data)

prep_path = path + '/data/preprocessing/' + prep_version + '/'
config.dict['preprocessing']['path'][prep_version] = prep_path
config.save(json_path)

if prep_version not in os.listdir(path + '/data/preprocessing/'):
    os.mkdir(prep_path)
prep_data.to_csv(prep_path + data_name + '.csv', index = False)
