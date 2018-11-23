import json
import plotly
import pandas as pd
import re 

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine
import pickle 

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if bool(re.search(r"[^a-zA-Z0-9]", w)) != True]
    tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in tokens if w not in stopwords.words("english")]
    tokens = [PorterStemmer().stem(w) for w in tokens]

    return tokens

# load data
engine = create_engine('sqlite:///data/DisasterMessages.db')
df = pd.read_sql_table("messages", engine)

# load model
model = joblib.load("models/classifier.pkl")

# load the aggregated down text representation. 
with open("app/text_cluster.pkl", "rb") as infile:
    text_cluster = pickle.load(infile)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Counts of different disasters
    df_categories = df.drop(["id", "message", "original", "genre"], axis=1)
    disaster_counts = [df[column].value_counts()[1] if 1 in df[column].unique() else 0 for column in df_categories.columns]
    disaster_counts_dict = dict(zip(df_categories.columns, disaster_counts))
    disaster_counts_dict = sorted(disaster_counts_dict.items(), key=lambda x: x[1], reverse=True)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=[disaster[0] for disaster in disaster_counts_dict],
                    y=[disaster[1] for disaster in disaster_counts_dict]
                )
            ],

            'layout': {
                'title': 'Distribution of aid required',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Aid"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x = [item[0] for item in list(text_cluster.values())],
                    y = [item[1] for item in list(text_cluster.values())],
                    mode = 'markers',
                    text = [item for item in list(text_cluster.keys())],
                    marker = dict(
                                size = 10,
                                color = 'rgba(152, 0, 0, .8)',
                                line = dict(
                                    width = 2,
                                    color = 'rgb(0, 0, 0)'
                            )
    )
                )
            ],
            'layout': {
                'title': 'Messages plotted using BOW features',
                'yaxis': {
                    'title': "Aggregated Component 1"
                },
                'xaxis': {
                    'title': "Aggregated Component 2"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query') 
    print(query)
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()