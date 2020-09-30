
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import config

class NERData():
    def __init__(self):
        self.data = None
        self.words = None
        self.tags = None
        self.sentences = None
        self.vocab_to_int = None
        self.tag_to_int = None

    def load_data(self):
        self.data = pd.read_csv('./NER/input/ner_dataset.csv',encoding = 'latin1')
        self.data = self.data.fillna(method='ffill')
        print("Unique Words ", self.data['Word'].nunique())
        print("Unique Tags ", self.data['Tag'].nunique())
        self.words = list(set(self.data['Word'].values))
        self.tags = list(set(self.data['Tag'].values))
        self.words.append('PAD')


    def sentence_getter(self):
        agg_func = lambda s:[(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                          s["POS"].values.tolist(),
                                                          s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

        return self.sentences

    def vocab_dict(self):
        self.vocab_to_int = {w:i+1 for i,w in enumerate(self.words)}
        self.int_to_vocab = {i: w for w, i in self.vocab_to_int.items()}
        self.tag_to_int = {w:i for i,w in enumerate(self.tags)}
        self.int_to_tag = {i: w for w, i in self.tag_to_int.items()}
        return self.vocab_to_int, self.tag_to_int

    def encode_text(self):
        self.encoded_sentence = []
        self.encoded_tag = []
        for sentence in self.sentences:
            self.encoded_sentence.append([self.vocab_to_int[w[0]] for w in sentence])
            self.encoded_tag.append([self.tag_to_int[w[2]] for w in sentence])

    def pad_features(self):

        self.padded_sentence = np.zeros((len(self.sentences), config.SEQ_LENGTH),dtype=int)
        self.padded_tag = np.zeros((len(self.sentences), config.SEQ_LENGTH),dtype=int)

        print("Padding Sentence")
        for i, row in enumerate(self.encoded_sentence):
            self.padded_sentence[i, -len(row):] = np.array(row)[:config.SEQ_LENGTH]
        print("Padding Tag")
        for i, row in enumerate(self.encoded_tag):
            self.padded_tag[i, -len(row):] = np.array(row)[:config.SEQ_LENGTH]

    def process_text(self, text):
        encoded_text = []
        for word in text.split():
            code = self.vocab_to_int.get(word)
            if code != None:
                encoded_text.append(code)

        padded_text = np.zeros((1, config.SEQ_LENGTH),dtype=int)
        padded_text[0,-len(encoded_text):] = encoded_text

        return padded_text


# plt.hist([len(s) for s in sentences],bins=50)
# plt.show()



