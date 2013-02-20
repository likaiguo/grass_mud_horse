import data_io
from features import FeatureMapper, SimpleTransform
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def feature_extractor(data):
    features = [('FullDescription-Bag of Words', 'FullDescription', CountVectorizer(max_features=200)),
                ('Title-Bag of Words', 'Title', CountVectorizer(max_features=100)),
                ('LocationRaw-Bag of Words', 'LocationRaw', CountVectorizer(max_features=100)),
                ('LocationNormalized-Bag of Words', 'LocationNormalized', CountVectorizer(max_features=100)),
                ('Category-Bag of Words','Category',CountVectorizer(max_features=100)),
                ('Company-Bag of Words','Company',CountVectorizer(max_features=100)),
                ('ContractType-Bag of Words','ContractType',CountVectorizer(tokenizer=my_tokenizer,min_df=1)),
                ('ContractTime-Bag of Words','ContractTime',CountVectorizer(tokenizer=my_tokenizer,min_df=1)),
                ('Source-Bag of Words','SourceName',CountVectorizer(tokenizer=my_tokenizer,min_df=1))]
    combined = FeatureMapper(features)
    return combined

#callable that provides tokens for Company, ContractType, ContractTime and Source
def my_tokenizer(s):
    return s.split('?')


def get_pipeline(data):
    features = feature_extractor(data)
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=30, 
                                                verbose=2,
                                                n_jobs=4,
                                                min_samples_split=20,
                                                random_state=3465343))]
    # steps = [("extract_features", features),
    #          ("classify", SVC(verbose=True))]

    return Pipeline(steps)

def main():
    print("Reading in the training data")
    train = data_io.get_train_df()

    print("Extracting features and training model")
    for key in train:
        classifier = get_pipeline(train[key])
        classifier.fit(train[key], train[key]["SalaryNormalized"])
        print("Saving the classifier for %s" %key)
        data_io.save_model(classifier,key)
    
if __name__=="__main__":
    main()