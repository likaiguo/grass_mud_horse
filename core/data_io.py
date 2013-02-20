import csv
import json
import numpy as np
import os
import pandas as pd
import pickle

def get_paths():
    paths = json.loads(open("Settings.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def identity(x):
    return x

def my_tokenizer(s):
    return s.split('?')


# For pandas >= 10.1 this will trigger the columns to be parsed as strings
converters = { "FullDescription" : identity
             , "Title": identity
             , "LocationRaw": identity
             , "LocationNormalized": identity
             }

def get_train_df():
    train_path = get_paths()["train_data_path"]
    train=pd.read_csv(train_path, converters=converters)
    for col in ['Company','ContractType','ContractTime','SourceName']:
        train[col][train[col]!=train[col]]=''
    all_types=pd.unique(train['Category']).tolist()
    T={}
    minor=['Part time Jobs','Domestic help & Cleaning Jobs']
    for type in all_types:
        if type not in minor:
            T[type]=train[train['Category']==type]
    T['Part time & Domestic help & Cleaning Jobs']=train[map(lambda s: s in minor, train['Category'])]
    return T
    

def get_valid_df():
    valid_path = get_paths()["valid_data_path"]
    valid=pd.read_csv(valid_path, converters=converters)
    for col in ['Company','ContractType','ContractTime','SourceName']:
        valid[col][valid[col]!=valid[col]]=''
    all_types=pd.unique(valid['Category']).tolist()
    V={}
    minor=['Part time Jobs','Domestic help & Cleaning Jobs']
    for type in all_types:
        if type not in minor:
            V[type]=valid[valid['Category']==type]
    V['Part time & Domestic help & Cleaning Jobs']=valid[map(lambda s: s in minor, valid['Category'])]
    return V

def save_model(model,name):
    if name=='Other/General Jobs':
        name='Other_General Jobs'
    out_path = get_paths()["model_path"]+name+'.pickle'
    pickle.dump(model, open(out_path, "w"))

def load_model(name):
    if name=='Other/General Jobs':
        name='Other_General Jobs'
    in_path = get_paths()["model_path"]+name+'.pickle'
    return pickle.load(open(in_path))

def write_submission(predictions):
    prediction_path = get_paths()["prediction_path"]
    writer = csv.writer(open(prediction_path, "w"), lineterminator="\n")
    valid = get_valid_df()
    writer.writerow(("Id", "SalaryNormalized"))
    for key in valid:
        rows = [x for x in zip(valid[key]["Id"], predictions[key].flatten())]
        writer.writerows(rows)