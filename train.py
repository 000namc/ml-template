
import os
import json
import sys
import pickle

# os.chdir('../')
path = os.getcwd() 
sys.path.insert(0, path)
sys.path.insert(0, path + '/model/')

json_path = path + '/data/config.json'
json_pars = path + '/model/model_pars.json'

from utils import Config
config = Config(json_path)
model_pars = Config(json_pars)
import model
import pandas as pd

####################################################
####################################################

prep_version = sys.argv[1]
model_version = sys.argv[2]
par_version = sys.argv[3]

# prep_version = 'basic'
# model_version = 'logistic'
# par_version = 'basic'

####################################################
####################################################

prep_path = path + '/experiments/' + prep_version + '/'
if prep_version not in os.listdir(path + '/experiments/'):
    os.mkdir(prep_path)
    
model_path = prep_path + model_version + '/'
if model_version not in os.listdir(prep_path):
    os.mkdir(model_path)
    
par_path = model_path + par_version + '/'
if par_version not in os.listdir(model_path):
    os.mkdir(par_path)

pars = model_pars.dict[model_version][par_version]
ml_model = getattr(sys.modules['model'].ml_models,model_version)(pars)

# ml_model.fit()
# 
