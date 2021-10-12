import re
import pandas as pd
import nltk
import spacy
import pymorphy2
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

nltk.download('wordnet')
nltk.download('stopwords')
morph = pymorphy2.MorphAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words_rus = set(stopwords.words('russian'))
stop_words_en = set(stopwords.words('english'))
stemmer_rus = SnowballStemmer('russian')
stemmer_en = SnowballStemmer('english')


class TextPreprocessor:
    def __init__(self, config):
        """Preparing text features.
        """
        self._mode_lower = config.get('mode_lower', True)
        self._mode_del_numbers = config.get('mode_del_numbers', True)
        self._mode_stemming = config.get('mode_stemming', True)
        self._mode_lemma = config.get('mode_norm', True)
        self._mode_remove_stops = config.get('mode_remove_stops', True)
        self._mode_drop_long_words = config.get('mode_drop_long_words', True)
        self._mode_drop_short_words = config.get('mode_drop_short_words', True)
        self._min_len_word = config.get('min_len_word', 3)
        self._max_len_word = config.get('max_len_word', 17)
        self._text_column = config.get('text_column', str)
        self._mode_save_df = config.get('mode_save_df', True)
        self._mode_del_eng = config.get('mode_del_eng', True)
        self._mode_del_punctuation = config.get('mode_del_punctuation', False)
        self._mode_sent_split = config.get('mode_sent_split', True)

    def _clean_text(self, input_text):
        """Delete special symbols."""
        if self._mode_lower:
            # Make text lower
            input_text = input_text.str.lower()

        if self._mode_del_numbers:
            # Delete numbers
            input_text = input_text.str.replace(r'[0-9]+', ' ')

        if self._mode_del_eng:
            # Delete english words
            input_text = input_text.str.replace(r'[A-Za-z]+', ' ')

        if self._mode_del_punctuation:
            # Delete punctuation
            input_text = input_text.str.replace(r'[^A-Za-zА-Яа-я0-9- ]+', ' ')

        # Delete special symbols
        input_text = input_text.str.replace(r'\s+', ' ')
        input_text = input_text.str.replace(r' +', ' ')
        input_text = input_text.str.replace(r'^ ', '')
        input_text = input_text.str.replace(r' $', '')

        return input_text

    def _text_lemmatization_rus(self, input_text):
        """Lemmatization of russian text"""
        return ' '.join([morph.parse(item)[0].normal_form for item in input_text.split(' ')])

    def _text_lemmatization_en(self, input_text):
        """lemmatization of english text"""
        return ' '.join([lemmatizer.lemmatize(item) for item in input_text.split(' ')])

    def _remove_stops_rus(self, input_text):
        """Delete russian stop-words"""
        return ' '.join([w for w in input_text.split() if not w in stop_words_rus])

    def _remove_stops_en(self, input_text):
        """Delete english stop-words"""
        return ' '.join([w for w in input_text.split() if not w in stop_words_en])

    def _stemming_rus(self, input_text):
        """Stemming of russian text"""
        return ' '.join([stemmer_rus.stem(item) for item in input_text.split(' ')])

    def _stemming_en(self, input_text):
        """Stemming of english text"""
        return ' '.join([stemmer_en.stem(item) for item in input_text.split(' ')])

    def _drop_long_words(self, input_text):
        """Delete long words"""
        return ' '.join([item for item in input_text.split(' ') if len(item) < self._max_len_word])

    def _drop_short_words(self, input_text):
        """Delete short words"""
        return ' '.join([item for item in input_text.split(' ') if len(item) > self._min_len_word])

    def transform(self, df):

        df[self._text_column] = df[self._text_column].astype('str')
        df['clean_text'] = df[self._text_column]

        # if self._del_orig_col:
        #     df = df.drop(self._columns_names, 1)

        if self._mode_sent_split:
            # Split by sentences
            df_seq = pd.DataFrame(columns=['clean_text'])

            for i in tqdm(range(df.shape[0])):
                df_temp = pd.DataFrame({
                    self._text_column: re.split('\. |\!|\?', df.loc[df.index[i], self._text_column]),
                    'clean_text': re.split('\. |\!|\?',df.loc[df.index[i], 'clean_text']),
                })
                df_seq = pd.concat([df_seq, df_temp])

            df_seq.index = range(df_seq.shape[0])
            df = df_seq[(df_seq['clean_text'] != ' ') | (df_seq['clean_text'] != '')]

        # Clean text
        df['clean_text'] = self._clean_text(df['clean_text'])

        if self._mode_lemma:
            # Lemmatization rus words
            df['clean_text'] = df['clean_text'].apply(self._text_lemmatization_rus, 1)
            # Lemmatization eng words
            df['clean_text'] = df['clean_text'].apply(self._text_lemmatization_en, 1)

        if self._mode_remove_stops:
            # Remove rus stop-words
            df['clean_text'] = df['clean_text'].apply(self._remove_stops_rus, 1)
            # Remove eng stop-words
            df['clean_text'] = df['clean_text'].apply(self._remove_stops_en, 1)

        if self._mode_stemming:
            # Stemming rus words
            df['clean_text'] = df['clean_text'].apply(self._stemming_rus, 1)
            # Stemming eng words
            df['clean_text'] = df['clean_text'].apply(self._stemming_en)

        if self._mode_drop_long_words:
            # Delete long words
            df['clean_text'] = df['clean_text'].apply(self._drop_long_words, 1)

        if self._mode_drop_short_words:
            # Delete short words
            df['clean_text'] = df['clean_text'].apply(self._drop_short_words, 1)

        df.loc[(df.clean_text == ''), ('clean_text')] = '<EMPT>'

        df = df[(df.clean_text != '<EMPT>')]

        return df


def build_vocabulary(df, min_count):
    """Build vocabulary"""
    our_vocab = []
    for i in df.index:
        our_vocab.extend([item for item in df.loc[i, 'clean_text'].split(' ')])
    our_vocab = pd.Series(our_vocab)
    our_vocab = our_vocab.value_counts()
    return our_vocab[our_vocab > min_count]


def add_pos_tags(our_vocab, POS_MODEL):
    nlp = spacy.load(POS_MODEL)
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    nlp.max_length = 30000000
    string = ' '.join(our_vocab.index)
    doc = nlp(string, disable = ['ner', 'parser'])
    for s in doc.sents:
        string = ' '.join([t.text+'_'+t.pos_ for t in s])
    return string.split(' ')
