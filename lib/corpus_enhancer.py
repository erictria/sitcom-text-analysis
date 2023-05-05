# Eric Tria (emt4wf@virginia.edu) DS 5001 Spring 2023

import pandas as pd
import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

class CorpusEnhancer:
    '''
    A class that adds additional details to CORPUS and VOCAB tables
    
    Code for these functions are based off of the code provided by Prof. Rafael Alvarado for DS 5001
    '''

    def __init__(self, corpus, ohco):
        '''
        Purpose: Initiates the class
        
        INPUTS:
        corpus - Pandas dataframe of tokens
        ohco - List of OHCO level to use
        '''
        self.TOKENS = corpus
        self.OHCO = ohco
    
    def enhance_corpus(self, strip_hyphens, strip_whitespace, use_nltk):
        '''
        Purpose: Add more features to tokens corpus
        
        INPUTS:
        strip_hyphens - boolean flag to strip hyphens
        strip_whitespace - boolean flag to strip whitespace
        use_nltk - boolean flag to use nltk specific functions
        
        OUTPUTS:
        tokens - Pandas dataframe of tokens
        '''
        # use ohco here
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
        self.TOKENS['token_str'] = self.TOKENS.pos_tuple.apply(lambda x: x[0])
        self.TOKENS['term_str'] = self.TOKENS.token_str.str.lower() 

        if not use_nltk:
            self.TOKENS['term_str'] = self.TOKENS.token_str.str.replace(r'[\W_]+', '', regex=True).str.lower()  
        else:
            punc_pos = ['$', "''", '(', ')', ',', '--', '.', ':', '``']
            self.TOKENS['term_str'] = self.TOKENS[~self.TOKENS.pos.isin(punc_pos)].token_str\
                .str.replace(r'[\W_]+', '', regex=True).str.lower()
        
        self.TOKENS = self.TOKENS[self.TOKENS.term_str != '']
        self.TOKENS = self.TOKENS.reset_index().set_index(self.OHCO)[['pos_tuple', 'pos', 'token_str', 'term_str']]

        return self.TOKENS

    def generate_enhanced_vocab(self, ohco, tf_method = 'sum'):
        '''
        Purpose: Enhance VOCAB table by adding metrics related to TFIDF
        
        INPUTS:
        ohco - list of OHCO level used
        tf_method - string method used for computing TFIDF
        
        OUTPUS:
        vocab - Pandas dataframe of enhanced vocab features
        '''
        self.VOCAB = self.TOKENS.term_str.value_counts().to_frame('n').sort_index()
        self.VOCAB.index.name = 'term_str'
        self.VOCAB['n_chars'] = self.VOCAB.index.str.len()
        self.VOCAB['p'] = self.VOCAB.n / self.VOCAB.n.sum()
        self.VOCAB['i'] = -np.log2(self.VOCAB.p)

        # self.VOCAB['s'] = 1 / self.VOCAB['p']
        # self.VOCAB['i'] = np.log2(self.VOCAB['s']) # Same as negative log probability (i.e. log likelihood): VOCAB['i'] = -np.log2(VOCAB.p)

        self.VOCAB['h'] = self.VOCAB['p'] * self.VOCAB['i']
        # self.H = self.VOCAB['h'].sum()

        # Get the most frequently associated part-of-space category for each word.
        self.VOCAB['max_pos'] = self.TOKENS[['term_str','pos']].value_counts().unstack(fill_value=0).idxmax(1)

        # Compute POS ambiguity
        self.VOCAB['n_pos'] = self.TOKENS[['term_str','pos']].value_counts().unstack().count(1)
        self.VOCAB['cat_pos'] = self.TOKENS[['term_str','pos']].value_counts().to_frame('n').reset_index()\
            .groupby('term_str').pos.apply(lambda x: set(x))
        
        # Add stopwords
        # We use NLTK's built in stopword list for English
        sw = pd.DataFrame(nltk.corpus.stopwords.words('english'), columns=['term_str'])
        sw = sw.reset_index().set_index('term_str')
        sw.columns = ['dummy']
        sw.dummy = 1

        self.VOCAB['stop'] = self.VOCAB.index.map(sw.dummy)
        self.VOCAB['stop'] = self.VOCAB['stop'].fillna(0).astype('int')

        # Add stems
        stemmer1 = PorterStemmer()
        self.VOCAB['stem_porter'] = self.VOCAB.apply(lambda x: stemmer1.stem(x.name), 1)

        stemmer2 = SnowballStemmer('english')
        self.VOCAB['stem_snowball'] = self.VOCAB.apply(lambda x: stemmer2.stem(x.name), 1)

        stemmer3 = LancasterStemmer()
        self.VOCAB['stem_lancaster'] = self.VOCAB.apply(lambda x: stemmer3.stem(x.name), 1)

        # Compute TFIDF and DFIDF
        BOW = self.__create_bow(ohco)

        TFIDF, DFIDF = self.__compute_tfidf_dfidf(tf_method)

        self.BOW['tfidf'] = self.TFIDF.stack()

        # Apply aggregates to VOCAB
        self.VOCAB['tfidf_mean'] = self.BOW.groupby('term_str').tfidf.mean() #TFIDF[TFIDF > 0].mean().fillna(0) # EXPLAIN
        self.VOCAB['tfidf_sum'] = self.TFIDF.sum()
        self.VOCAB['tfidf_median'] = self.BOW.groupby('term_str').tfidf.median() #TFIDF[TFIDF > 0].median().fillna(0) # EXPLAIN
        self.VOCAB['tfidf_max'] = self.TFIDF.max()

        self.VOCAB['dfidf'] = self.DFIDF

        return self.VOCAB
    
    def __create_bow(self, ohco):
        '''
        PURPOSE: generate a bag-of-words (BOW) dataframe given a corpus and bag level.
        
        INPUTS:
        tokens_df - a dataframe representing a corpus of tokens and terms.
        bag_choice - The bag level used for generating the BOW. A list of str that is a subset of
            ['book_id', 'chap_id', 'para_num', 'sent_num', 'token_num']
        
        OUTPUTS:
        BOW - a dataframe representing the bag-of-words (BOW).
        '''
        self.BOW = self.TOKENS.groupby(ohco + ['term_str'])\
            .term_str\
            .count()\
            .to_frame('n') 
        
        return self.BOW
    
    def __compute_tfidf_dfidf(self, tf_method = 'sum'):
        '''
        PURPOSE: compute the TFIDF values given a bag-of-words (BOW) dataframe and the TF method
        
        INPUTS:
        bow_df - The bag-of-words (BOW) dataframe.
        tf_method - The TF method is a str that should be one of the following:
            ['sum', 'max', 'log', 'raw', 'double_norm', 'binary']
        
        OUTPUTS:
        TFIDF - Term Frequency Inverse Document Frequency 
        '''
        
        # We create a document-term count matrix simply by unstacking the BOW
        # which converts it from a narrow to a wide representation. 
        DTCM = self.BOW.n.unstack().fillna(0).astype('int')
        
        # Compute TF
        print('Using TF method:', tf_method)
        if tf_method == 'sum':
            TF = DTCM.T / DTCM.T.sum()
        elif tf_method == 'max':
            TF = DTCM.T / DTCM.T.max()
        elif tf_method == 'log':
            TF = np.log2(1 + DTCM.T)
        elif tf_method == 'raw':
            TF = DTCM.T
        elif tf_method == 'double_norm':
            TF = DTCM.T / DTCM.T.max()
        elif tf_method == 'binary':
            TF = DTCM.T.astype('bool').astype('int')
        TF = TF.T
        
        # Compute DF
        DF = DTCM.astype('bool').sum()
        
        # Compute IDF
        # Using standard IDF method
        N = DTCM.shape[0]
        IDF = np.log2(N / DF)
        
        # Compute TFIDF
        self.TFIDF = TF * IDF
        
        # COMPUTE DFIDF
        self.DFIDF = DF * IDF
        
        return self.TFIDF, self.DFIDF