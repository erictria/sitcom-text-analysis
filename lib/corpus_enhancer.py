import pandas as pd
import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

class CorpusEnhancer:
    '''
    A class that adds additional details to CORPUS and VOCAB tables
    '''

    def __init__(self, corpus, vocab, ohco):
        self.TOKENS = corpus
        self.VOCAB = vocab
        self.OHCO = ohco
    
    def enhance_corpus(self, strip_hyphens, strip_whitespace):
        if strip_hyphens == True:
            self.TOKENS.token_str = self.TOKENS.token_str.str.replace(r"-", ' ')
        if strip_whitespace == True:
            self.TOKENS = self.TOKENS.token_str\
                    .apply(lambda x: pd.Series(
                            nltk.pos_tag(nltk.WhitespaceTokenizer().tokenize(x)),
                            dtype='object'
                        )
                    )
        else:
            self.TOKENS = self.TOKENS.token_str\
                    .apply(lambda x: pd.Series(nltk.pos_tag(nltk.word_tokenize(x))))
        
        self.TOKENS = self.TOKENS.stack().to_frame('pos_tuple')
        self.TOKENS['pos'] = self.TOKENS.pos_tuple.apply(lambda x: x[1])
        # self.TOKENS['token_str'] = self.TOKENS.pos_tuple.apply(lambda x: x[0])
        # self.TOKENS['term_str'] = self.TOKENS.token_str.str.lower() 

        return self.TOKENS

    def enhance_vocab(self):
        self.VOCAB['p'] = self.VOCAB.n / self.VOCAB.n.sum()
        self.VOCAB['n_chars'] = self.VOCAB.index.str.len()
        self.VOCAB['p'] = self.VOCAB['n'] / self.VOCAB['n'].sum()
        self.VOCAB['s'] = 1 / self.VOCAB['p']
        self.VOCAB['i'] = np.log2(self.VOCAB['s']) # Same as negative log probability (i.e. log likelihood): VOCAB['i'] = -np.log2(VOCAB.p)
        self.VOCAB['h'] = self.VOCAB['p'] * self.VOCAB['i']

        # self.H = self.VOCAB['h'].sum()

        return self.VOCAB