import sys
sys.path.append('..')
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from modules.text_preprocessor import TextPreprocessor


class TextSummarizer:
    """Text summarization.
    """
    def __init__(self, config):
        self._len_summary = config.get('len_summary', 50)
        self._text_column = config.get('text_column', 'text')

    def _create_tfidf_vocabulary(self, df):
        """"Generate a vacabulary with TFIDF-weights"""

        vectorizer = CountVectorizer()
        df_tfidf = vectorizer.fit_transform(df['clean_text'])

        vocabulary = pd.DataFrame({
            'words': vectorizer.get_feature_names(),
            'weights': df_tfidf.sum(0).tolist()[0]
        })

        vocabulary = vocabulary.sort_values('weights', ascending=False)

        return vocabulary

    def _calculate_sentence_weight(self, df, vocabulary):
        df['weight'] = 0
        for item in tqdm(range(df.shape[0])):
            words_sent = df['clean_text'].values[item].split(' ')
            weight = vocabulary[vocabulary['words'].isin(words_sent)]['weights'].sum()
            df.loc[df.index[item], 'weight'] = weight
        df = df.sort_values('weight', ascending=False)
        return df

    def transform(self, df):
        config = {
            'mode_lower': True,
            'mode_del_numbers': True,
            'mode_norm': True,
            'mode_stemming': False,
            'mode_remove_stops': True,
            'mode_drop_long_words': True,
            'mode_drop_short_words': False,
            'min_len_word': 3,
            'max_len_word': 25,
            'text_column': self._text_column,
            'mode_del_eng': True,
            'mode_del_punctuation': True,
            'mode_sent_split': True
        }
        df = TextPreprocessor(config).transform(df)
        df = df[df[self._text_column].str.len() < df[self._text_column].str.len().mean()]
        vocabulary = self._create_tfidf_vocabulary(df)
        df = self._calculate_sentence_weight(df, vocabulary)

        return '. '.join(df[self._text_column].values[0:self._len_summary])