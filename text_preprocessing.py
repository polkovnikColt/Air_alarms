import re
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from num2words import num2words

nltk.download('punkt')


def remove_one_letter_word(data):
    words = word_tokenize(str(data))

    new_text = ""
    for word in words:
        if len(word) > 1:
            new_text = new_text + " " + word
    return new_text


def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    stop_stop_words = {"no", "not"}
    stop_words = stop_words - stop_stop_words

    words = word_tokenize(str(data))

    new_text = ""
    for word in words:
        if word not in stop_words and len(word) > 1:
            new_text = new_text + " " + word
    return new_text


def remove_punctuation(data):
    symbols = "!\"#$%^&*—()_-=+@:;?<>`{|}[\]~\n"

    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for token in tokens:
        new_text = new_text + " " + stemmer.stem(token)
    return new_text


def lemmatizing(data):
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for token in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(token)
    return new_text


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for token in tokens:
        if token.isdigit():
            if int(token) < 100000000000:
                token = num2words(token)
            else:
                token = ''
        new_text = new_text + " " + token
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def remove_url_from_string(data):
    words = word_tokenize(str(data))

    new_text = ""
    for word in words:
        word = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(word), flags=re.MULTILINE)
        word = re.sub(r'^http?:\/\/.*[\r\n]*', '', str(word), flags=re.MULTILINE)

        new_text = new_text + " " + word
    return new_text