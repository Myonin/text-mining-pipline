import numpy as np
from tqdm import tqdm


class USEEncoder(object):
    """Generate embeddings from text with Universal Sentence Encoder.

    Parameters
    ----------
    config : dict
        Dictionary with configs.

    Attributes
    ----------
    _batch_size : int, default 16
        Size of a batch.
    _text_column : str, default 'text'
        Name of a text column in the dataframe.

    """
    def __init__(self, config):
        self._batch_size = config.get('batch_size', 16)
        self._text_column = config.get('text_column', 'text')

    def transform(self, df, use_model):
        """Short summary.

        Parameters
        ----------
        df : pandas dataframe
            DataFrame with a text column.
        use_model : TF model
            Pretrained Universal Sentence Encoder model.

        Returns
        -------
        numpy array
            Embeddings generated from text.
        """
        # Generating embeddings
        batch_range = np.arange(1, df.shape[0], self._batch_size)
        start_last_batch = batch_range[-1]
        end_last_batch = df.shape[0]+1
        batch_range = np.delete(batch_range, -1)

        # Generate embedding of the first row for array initialization.
        embeddings = use_model(df[self._text_column].values[0])
        for batch in tqdm(batch_range):
            embedding = use_model(df[self._text_column].values[batch:batch+self._batch_size])
            embeddings = np.vstack((embeddings, embedding))
        # Generate embeddings for the last batch.
        embedding = use_model(df[self._text_column].values[start_last_batch:end_last_batch])
        embeddings = np.vstack((embeddings, embedding))

        return embeddings
