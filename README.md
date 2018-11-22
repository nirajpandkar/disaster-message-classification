# Disaster Response Pipeline

> An NLP pipeline to clean, preprocess and train a text classifier for disaster response management. 

## File Structure

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py        # etl pipeline
|- DisasterMessages.db   # database to save clean data to

- models
|- train_classifier.py  # train the model
|- model-adaboost.pkl  # saved model 
```

## Installation

1. Install pip
```
sudo apt-get install python-pip
```
2. Install virtualenv
```
sudo pip install virtualenv
```
3. Create a virtualenv for this project
```
virtualenv -p python3 venv
```

## Usage

After cloning this repo, change directory into it and follow the instructions below - 

### Activate the virtualenv and install dependencies

```
source venv/bin/activate
pip install -r requirements.txt
```

### To run ETL pipeline (Cleans data and stores in database)

```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

### To run ML pipeline (Trains classifier and saves the model)

```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

**Note: Takes about 1 hour 40 minutes to finish training on a 6 core i5 8th gen CPU**

### To run the webapp

```
python app/run.py
```