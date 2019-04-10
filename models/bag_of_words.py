from models.base_model import BaseModel
import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
import sklearn.model_selection
from sklearn import svm
import unicodedata
import re
import pandas as pd
import joblib
import os
import time


class BagOfWordsModel(BaseModel):
    """Naive model which simply counts the word occurences in a sentence and then trains a SVC on it. 
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.vectorizer = None
        self.label_mapping = None
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=logging.INFO)

    def train(self, config):
        # config
        max_features = int(config.get('max_features', 1e4))
        max_df = config.get('max_df', 1)
        min_df = config.get('min_df', 0)
        ngrams = config.get('ngrams', 3)
        stop_words = config.get('stop_words', 'english')
        stop_words = config.get('stop_words', None)
        # read and transform data
        df_train = pd.read_csv(config.train_data)
        df_test = pd.read_csv(config.test_data)
        df = pd.concat([df_train, df_test])
        # compute mapping
        self.label_mapping = self.set_label_mapping(config)
        # train vectorizer
        print('Build vectorizer...')
        self.vectorizer = CountVectorizer(max_features=max_features, max_df=max_df, stop_words=stop_words, min_df=min_df, ngram_range=(1, ngrams))
        self.vectorizer = self.vectorizer.fit(df['text'])
        vectors, labels = self.vectorize_data(df_train)
        features = self.vectorizer.get_feature_names()
        if config.verbose:
            print("Num features: {}".format(len(features)))
        if len(features) < 1:
            raise Exception('Vecotrizer could not find any features under the given parameters. Try less restrictive parameters.')
        with open(os.path.join(config.output_path, 'vectorizer.pkl'), 'wb') as f:
            params = {**self.vectorizer.get_params(), 'vocabulary': features}
            joblib.dump(params, f)
        # train
        print('Start training...')
        t_start = time.time()
        self.model = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
        self.model.fit(vectors, labels)
        t_end = time.time()
        print('... training finished after {:.1f} min'.format((t_end - t_start)/60))
        print('Saving model...')
        with open(os.path.join(config.output_path, 'model.pkl'), 'wb') as f:
            joblib.dump(self.model, f)

    def test(self, config):
        self.load_model(config)
        # read data
        df = pd.read_csv(config.test_data)
        vectors, labels = self.vectorize_data(df)
        # predict
        predicted = self.model.predict(vectors)
        probabilities = self.model.predict_proba(vectors)
        metrics = self.performance_metrics(labels, predicted, label_mapping=self.get_label_mapping(config))
        return metrics

    def predict(self, config, data):
        self.load_model(config)
        df = pd.DataFrame({'text': data})
        vectors, _ = self.vectorize_data(df)
        probabilities = self.model.predict_proba(vectors)
        return self.format_predictions(probabilities, label_mapping=self.label_mapping)

    def load_model(self, config):
        # read model
        with open(os.path.join(config.output_path, 'model.pkl'), 'rb') as f:
            self.model = joblib.load(f)
        # read vectorizer
        with open(os.path.join(config.output_path, 'vectorizer.pkl'), 'rb') as f:
            vectorizer_params = joblib.load(f)
        self.vectorizer = CountVectorizer()
        self.vectorizer.set_params(**vectorizer_params)
        if self.label_mapping is None:
            self.label_mapping = self.get_label_mapping(config)

    def vectorize_data(self, df):
        df['text'] = df['text'].apply(lambda t: self.tokenize(t))
        vectors = self.vectorizer.transform(df['text'])
        if 'label' in df:
            labels = np.array(df['label'].apply(self.transform_labels))
        else:
            labels = None
        return vectors, labels

    def tokenize(self, text):
        # Replace unnecessary spacings/EOL chars
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        text = ' '.join(text)
        text = text.lower()
        # replace urls and mentions
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>', text)
        text = re.sub('(\@[^\s]+)','<user>', text)
        return text.strip()

    def transform_labels(self, label):
        return self.label_mapping[str(label)]

    def transform_index(self, ix):
        for i, j in self.label_mapping.items():
            if j == ix:
                return i
