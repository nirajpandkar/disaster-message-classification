import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories from csv files.

    Arguments:
        messages_filepath: File path to messages csv
        categories_filepath: File path to categories csv
    Return:
        df: A dataframe consisting of messages and categories merged.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how="inner", on="id")
    return df

def clean_data(df):
    """
    Cleans the dataframe by splitting categories into separate columns.

    Arguments:
        df: Dataframe consisting of messages and their categories. 
    Return:
        df: A clean dataframe with separate category columns.
    """

    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    
    # rename the columns of `categories`
    row = categories.loc[0].values
    category_colnames = [x.split("-")[0] for x in row]
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from df 
    # and concatenate the new categories
    # and drop duplicates
    df = df.drop(["categories"], axis = 1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates("message", keep="first")
    
    return df

def save_data(df, database_filename):
    """
    Saves the dataframe into sqlite database.

    Arguments:
        df: Dataframe consisting of messages and their categories.
        database_filename: Filepath to which the database should be saved.
    Return:
        None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()