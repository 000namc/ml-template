import pandas as pd
from sklearn.linear_model import LogisticRegression


class preprocess:

    def basic(data):
        prep_data = data.copy()
        return(prep_data)

    def etc(data):
        prep_data = data.copy()
        return(prep_data)
    
class ml_models:

    def logistic(pars):
        model = LogisticRegression(**pars)
        return(model)
