import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.framework import ops
import numpy as np
import tflearn
import tensorflow as tf
import random


from nltk import sent_tokenize

stemmer = LancasterStemmer()
nltk.download('punkt')


with open('data.json') as json_data:
    intents = json.load(json_data)
