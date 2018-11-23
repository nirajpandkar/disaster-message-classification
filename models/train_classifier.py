# import packages
import sys, os.path
import pickle
import string
import re

tokenizer_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '/app/')
sys.path.append(tokenizer_dir)

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
from tokenizer import tokenize

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

    X = df["message"].values
    y = df.loc[:, "related":"direct_report"]

    return X, y

def build_model():
    """
    Build a model pipeline.

    Arguments:
        None
    Returns:
        cv: A GridSearchCV object.
    """
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(lowercase=True, tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=0)))
    ])

    define parameters for GridSearchCV
    parameters = {
        'vect__ngram_range': ((1,1), (1,2)),
        'vect__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__learning_rate': [1.0, 1.5, 2.0]
    }

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, cv=2, n_jobs=6)
    print("Executing with 6 cores")
    return cv


def train(X, y, model):
    """
    Train the model by splitting dataset into training and testing datasets. 

    Arguments:
        X: Features.
        y: labels.
        model: A sklearn pipeline or GridSearchCV object.
    Returns:
        model: A model fit to the training dataset. 
    """
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # fit model
    model.fit(X_train, y_train)
    print("Model fit to training data!")

    return model

def evaluate(X, y, model):
    """
    Evaluate the model using metrics (Precision, Recall, F1 score)

    Arguments:
        X: Features.
        y: labels.
        model: A sklearn pipeline or GridSearchCV object.
    Returns:
        None.
    """
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    for i in range(36):
        pred = [x[i] for x in y_pred]
        true = [x[i] for x in y_test.values]
        print("\n\nClass: {}".format(y.columns[i]))
        print(classification_report(pred, true))
    
def export_model(model):
    """
    Save the model to disk.

    Arguments:
        model: A model fit to training dataset. 
    Returns:
        None. 
    """
    # Export model as a pickle file
    with open("models/classifier.pkl", "wb") as outfile:
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
    evaluate(X, y, model)


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
