from sklearn.model_selection import train_test_split
from gensim.models import FastText, fasttext
from tensorflow.python.keras.preprocessing import sequence
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def train_val_test_split(X_data, y_data, val_size, test_size):
    """
    Lemmatize english texts.
    
     # Arguments:
        X_data: list of the data
        y_data: list of the labels
        val_size: ratio of the validate set
        test_size: ratio of the text set
    # Returns:
        train, validate and text data sets, and labels for every data set
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=1)
    
    return X_train, X_val, X_test, y_train, y_test, y_val

def prepare_fastText_embedding_matrix(word_index, model_file):
    """
    Create an embedding matrix: n*m, where n is the number of words in word index
    m is the size of an embedding vector.
    
     # Arguments:
        word_index: dictionary keys are words, values are the word indexes
        model_file: path to the pretrained word embeddings
    # Returns:
        the embedding matrix
    """
    model = fasttext.load_facebook_vectors(model_file)
    EMBEDDING_DIM = model.vector_size
    embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = model[word]
        embedding_matrix[i-1] = embedding_vector
    
    return embedding_matrix

def text_to_sequence(text, vocabulary):
    sentence_array = list()
    for word in text.split():
        try:
            sentence_array.append(vocabulary[word])
        except KeyError:
            continue
    return np.array(sentence_array)

def sequence_vectorize(train_texts, val_texts, test_texts, max_features=10000, sequence_length = 10000):
    """Vectorizes texts as sequence vectors. If seqence_length is not None 1 text = 1 sequence vector
    with length sequence_length. If sequence_lenght is None the output vectors are variable-length. 
    
    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.
        test_texts: list, test text string.
        sequence_length: lenght of the sequences.
    # Returns
        x_train, x_val, x_test word_index: vectorized training, validation and test
            texts and word index dictionary.
    """
    cv = CountVectorizer(max_features=max_features)
    cv.fit_transform(train_texts)
    
    x_train = np.array([text_to_sequence(text, cv.vocabulary_) for text in train_texts])
    x_val = np.array([text_to_sequence(text, cv.vocabulary_) for text in val_texts])
    x_test = np.array([text_to_sequence(text, cv.vocabulary_) for text in test_texts])

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=sequence_length)
    x_val = sequence.pad_sequences(x_val, maxlen=sequence_length)
    x_test = sequence.pad_sequences(x_test, maxlen=sequence_length)
       
    return x_train, x_val, x_test, cv.vocabulary_