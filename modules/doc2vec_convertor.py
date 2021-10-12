from tqdm import tqdm
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


class Doc2VecConvertor(object):
    def __init__(self, config):
        self._text_column = config.get('text_column', 'text')
        self._embedding_size = config.get('embedding_size', 300)
        self._min_word = config.get('min_word', 5)
        self._window_size = config.get('window_size', 40)
        self._down_sampling = config.get('down_sampling', 1e-2)
        self._epochs = config.get('epochs', 100)

    def _doc2vec_encoder(self, df, model):
        # Doc2Vec is generating embeddings...
        # Generate embedding of the first row for array initialization.
        embeddings = model.infer_vector(df[self._text_column].values[0].split())
        for sentence in tqdm(df[self._text_column].values[1:]):
            embedding = model.infer_vector(sentence.split())
            embeddings = np.vstack((embeddings, embedding))
        return embeddings

    def fit(self, df):
        # Training Doc2Vec...
        train_tokenized = [TaggedDocument(sentence.split(), [n]) for n, sentence in enumerate(df[self._text_column].values)]
        vectorizer = Doc2Vec(
            vector_size=self._embedding_size,
            window=self._window_size,
            min_count=self._min_word,
            sample=self._down_sampling,
            epochs=self._epochs
            )
        vectorizer.build_vocab(documents=train_tokenized)
        vectorizer.train(
            documents=train_tokenized,
            total_examples=vectorizer.corpus_count,
            epochs=vectorizer.epochs
            )
        return vectorizer

    def transform(self, df, vectorizer):
        edembeddings = self._doc2vec_encoder(df, vectorizer)
        return edembeddings
