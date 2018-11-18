# import packages
import sys
import pickle
import string
import re

# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

def load_data(data_file):
    """
    Load the data from sqlite database.

    Arguments:
        data_file: SQLite database file path.
    Returns:
        X: Features
        y: labels
    """
    engine = create_engine('sqlite:///' + data_file)

    df = pd.read_sql_table("messages", engine)

    X = df["message"]
    y = df.loc[:, "related":"direct_report"]

    return X, y


def tokenize(text):
    """
    Generates tokens from a given document
    
    Arguments:
        text (str): Document to be tokenized
    
    Returns:
        tokens: A list of tokens
    """
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if bool(re.search(r"[^a-zA-Z0-9]", w)) != True]
    tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in tokens if stopwords.words("english")]
    tokens = [PorterStemmer().stem(w) for w in tokens]
    return tokens


def build_model():
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(lowercase=True, tokenizer=tokenize, stop_words=stopwords.words("english"))),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=0)))
    ])

    # # define parameters for GridSearchCV
    # parameters = {
    #     'vect__ngram_range': ((1,1), (1,2)),
    #     'vect__use_idf': (True, False),
    #     'clf__estimator__n_estimators': [50, 100, 200],
    #     'clf__estimator__learning_rate': [1.0, 1.5, 2.0]
    # }

    # # create gridsearch object and return as final model pipeline
    # cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=1)

    # return cv
    return pipeline


def train(X, y, model):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2, random_state=42)

    # fit model
    model.fit(X_train, y_train)
    print("Model fit to training data!")
    # output model test results
    evaluate(X, y)
    return model

def evaluate(X, y):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = None
    with open("baseline.pkl", "rb") as infile:
        model = pickle.load(infile)
    y_pred = model.predict(X_test)
    for i in range(36):
        pred = [x[i] for x in y_pred]
        true = [x[i] for x in y_test.values]
        print("\n\nClass: {}".format(y.columns[i]))
        print(classification_report(pred, true))
    
def export_model(model):
    # Export model as a pickle file
    with open("model-adaboost.pkl", "wb") as outfile:
        pickle.dump(model, outfile)



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    print("Loaded data!\n")
    model = build_model()  # build model pipeline
    print("Model built!\n")
    model = train(X, y, model)  # train model pipeline
    print("Model trained!\n")
    export_model(model)  # save model
    print("Model exported!")


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
