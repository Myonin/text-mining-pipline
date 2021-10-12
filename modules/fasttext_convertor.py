from tqdm import tqdm
import numpy as np
from gensim.models import FastText


class FastTextConvertor(object):
    def __init__(self, config):
        self._text_column = config.get('text_column', 'text')
        self._embedding_size = config.get('embedding_size', 300)
        self._min_word = config.get('min_word', 5)
        self._window_size = config.get('window_size', 40)
        self._down_sampling = config.get('down_sampling', 1e-2)
        self._epochs = config.get('epochs', 200)

    def _avg_feature_vector(self, sentence, model, num_features, index2word_set):
        words = sentence.split()
        feature_vec = np.zeros((num_features, ), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    def _fasttext_encoder(self, df, model):
        index2word_set = set(model.wv.index2word)
        # Generate embedding of the first row for array initialization.
        embeddings = self._avg_feature_vector(
            df[self._text_column].values[0],
            model,
            self._embedding_size,
            index2word_set
            )
        for sentence in tqdm(df[self._text_column].values[1:]):
            embedding = self._avg_feature_vector(
                sentence,
                model,
                self._embedding_size,
                index2word_set
                )
            embeddings = np.vstack((embeddings, embedding))
        return embeddings

    def fit(self, df):
        # Training FastText...
        train_tokenized = [sentence.split() for sentence in df[self._text_column].values]
        vectorizer = FastText(
            sentences=train_tokenized,
            size=self._embedding_size,
            window=self._window_size,
            min_count=self._min_word,
            sample=self._down_sampling,
            iter=self._epochs
            )
        return vectorizer

    def transform(self, df, vectorizer):
        # FastText is generating embeddings...
        embeddings = self._fasttext_encoder(df, vectorizer)
        return embeddings
