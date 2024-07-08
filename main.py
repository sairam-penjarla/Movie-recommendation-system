import numpy as np
import pandas as pd
import logging
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


# Load configuration from config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(config['logging']['log_file']), logging.StreamHandler()])

class DataLoader:
    def __init__(self, filepath, columns):
        self.filepath = filepath
        self.columns = columns
        self.df = None

    def preprocess_data(self):
        logging.info("Loading data from file: %s", self.filepath)
        self.df = pd.read_csv(self.filepath)
        logging.info("Preprocessing data")
        
        # Rename columns
        self.df = self.df.rename(columns={
            "listed_in"         : self.columns['genre'],
            "director"      : self.columns['director'],
            "cast"          : self.columns['cast'],
            "description"   : self.columns['description'],
            "title"         : self.columns['title'],
            "date_added"     : self.columns['date_added'],
            "country"       : self.columns['country']
        })
        
        # Fill missing values
        self.df[self.columns['country']] = self.df[self.columns['country']].fillna(self.df[self.columns['country']].mode()[0])
        self.df[self.columns['date_added']] = self.df[self.columns['date_added']].fillna(self.df[self.columns['date_added']].mode()[0])
        self.df[self.columns['rating']] = self.df[self.columns['rating']].fillna(self.df[self.columns['country']].mode()[0])
        self.df = self.df.dropna(how='any', subset=[self.columns['cast'], self.columns['director']])
        
        # Further processing
        self.df['category'] = self.df[self.columns['genre']].apply(lambda x: x.split(",")[0])
        self.df['YearAdded'] = self.df[self.columns['date_added']].apply(lambda x: x.split(" ")[-1])
        self.df['MonthAdded'] = self.df[self.columns['date_added']].apply(lambda x: x.split(" ")[0])
        self.df['country'] = self.df[self.columns['country']].apply(lambda x: x.split(",")[0])
        
        return self.df

class DataTransformer:
    def __init__(self, df):
        self.df = df
        self.features = ['category', 'director_name', 'cast_members', 'summary', 'movie_title']
        self.filters = self.df[self.features]

    @staticmethod
    def clean_text(text):
        return str.lower(text.replace(" ", ""))

    def apply_transformations(self):
        logging.info("Applying data transformations")
        for feature in self.features:
            self.filters[feature] = self.filters[feature].apply(self.clean_text)
        
        self.filters['Soup'] = self.filters.apply(self.create_soup, axis=1)
        return self.filters

    @staticmethod
    def create_soup(row):
        return f"{row['director_name']} {row['cast_members']} {row['category']} {row['summary']}"

class ModelTrainer:
    def __init__(self, filters):
        self.filters = filters
        self.count_vectorizer = CountVectorizer(stop_words='english')

    def train_model(self):
        logging.info("Training model")
        self.count_matrix = self.count_vectorizer.fit_transform(self.filters['Soup'])
        self.cosine_sim_matrix = cosine_similarity(self.count_matrix, self.count_matrix)
        return self.cosine_sim_matrix

class Recommender:
    def __init__(self, df, filters, cosine_sim_matrix):
        self.df = df
        self.cosine_sim_matrix = cosine_sim_matrix
        filters = filters.reset_index()
        self.indices = pd.Series(filters.index, index=filters['movie_title'])

    def get_recommendations(self, title):
        logging.info("Getting recommendations for title: %s", title)
        title = title.replace(' ', '').lower()
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return self.df['movie_title'].iloc[movie_indices]

# Usage
data_loader = DataLoader(config['data']['filepath'], config['data']['columns'])
df = data_loader.preprocess_data()

data_transformer = DataTransformer(df)
filters = data_transformer.apply_transformations()

model_trainer = ModelTrainer(filters)
cosine_sim_matrix = model_trainer.train_model()

recommender = Recommender(df, filters, cosine_sim_matrix)
print(recommender.get_recommendations('PK'))