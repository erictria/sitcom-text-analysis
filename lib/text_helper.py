# Eric Tria (emt4wf@virginia.edu) DS 5001 Spring 2023

import numpy as np
import pandas as pd

from numpy.linalg import norm

from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.manifold import TSNE

import nltk

import matplotlib.pyplot as plt

import seaborn as sns

from gensim.models import word2vec

import plotly_express as px

from sklearn.decomposition import PCA
from scipy.linalg import norm
from scipy.linalg import eigh
import plotly_express as px

from IPython.display import display, HTML

class TextHelper:
    '''
    A class that contains helper functions used to conduct various text analyses
    
    Code for these functions are based off of the code provided by Prof. Rafael Alvarado for DS 5001
    '''
    
    def __init__(self):
        '''
        Purpose: Initiate the class
        '''
        
        pass
    
    def create_bow(self, TOKENS, ohco):
        '''
        PURPOSE: generate a bag-of-words (BOW) dataframe given a corpus and bag level.
        
        INPUTS:
        tokens_df - a dataframe representing a corpus of tokens and terms.
        bag_choice - The bag level used for generating the BOW. A list of str that is a subset of
            ['book_id', 'chap_id', 'para_num', 'sent_num', 'token_num']
        
        OUTPUTS:
        BOW - a dataframe representing the bag-of-words (BOW).
        '''
        
        BOW = TOKENS.groupby(ohco + ['term_str'])\
            .term_str\
            .count()\
            .to_frame('n') 
        
        return BOW
    
    def compute_tfidf_dfidf(self, BOW, tf_method = 'sum'):
        '''
        PURPOSE: compute the TFIDF values given a bag-of-words (BOW) dataframe and the TF method
        
        INPUTS:
        bow_df - The bag-of-words (BOW) dataframe.
        tf_method - The TF method is a str that should be one of the following:
            ['sum', 'max', 'log', 'raw', 'double_norm', 'binary']
        
        OUTPUTS:
        TFIDF - Term Frequency Inverse Document Frequency 
        DFIDF - Document Frequency Inverse Document Frequency 
        '''
        
        # We create a document-term count matrix simply by unstacking the BOW
        # which converts it from a narrow to a wide representation. 
        DTCM = BOW.n.unstack().fillna(0).astype('int')
        
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
        TFIDF = TF * IDF
        
        # COMPUTE DFIDF
        DFIDF = DF * IDF
        
        return TFIDF, DFIDF
    
    # CLUSTERING FUNCTIONS
    
    def generate_clustering_pairs(self, TFIDF_COLLAPSED, LIB):
        '''
        Purpose: Generate clustering pairs using various distance metrics
        
        INPUTS:
        TFIDF_COLLAPSED - Pandas dataframe of TFIDF values
        LIB - Pandas dataframe of LIB info
        
        OUTPUTS:
        PAIRS - Pandas dataframe of clustering pairs
        '''
        
        L0 = TFIDF_COLLAPSED.astype('bool').astype('int') # Binary (Pseudo L)
        L1 = TFIDF_COLLAPSED.apply(lambda x: x / x.sum(), 1) # Probabilistic
        L2 = TFIDF_COLLAPSED.apply(lambda x: x / norm(x), 1) # Pythagorean / Euclidean

        PAIRS = pd.DataFrame(index=pd.MultiIndex.from_product([LIB.index.tolist(), LIB.index.tolist()])).reset_index()
        PAIRS = PAIRS[PAIRS.level_0 < PAIRS.level_1].set_index(['level_0','level_1'])
        PAIRS.index.names = ['doc_a', 'doc_b']
        
        PAIRS['cityblock'] = pdist(TFIDF_COLLAPSED, 'cityblock')
        PAIRS['cosine'] = pdist(TFIDF_COLLAPSED, 'cosine')
        PAIRS['euclidean'] = pdist(L2, 'euclidean')
        PAIRS['jaccard'] = pdist(L0, 'jaccard')
        PAIRS['js'] = pdist(L1, 'jensenshannon')
        
        return PAIRS
    
    def hca(self, LIB, sims, linkage_method='complete', color_thresh=.3, figsize=(10, 10)):
        '''
        Purpose: visualize clustering diagrams
        '''
        
        tree = sch.linkage(sims, method=linkage_method)
        labels = LIB.series_season.values ## edit this
        plt.figure()
        fig, axes = plt.subplots(figsize=figsize)
        dendrogram = sch.dendrogram(tree, 
                                    labels=labels, 
                                    orientation="left", 
                                    count_sort=True,
                                    distance_sort=True,
                                    above_threshold_color='.75',
                                    color_threshold=color_thresh
                                   )
        plt.tick_params(axis='both', which='major', labelsize=14)
    
    # PCA
    
    def get_pca_doc(self, CORPUS, VOCAB, LIB, OHCO):
        '''
        Purpose: Generate the DOC_SUM and TFIDF dataframes needed for PCA
        '''
        
        BOW_PCA = self.create_bow(CORPUS, OHCO)
        TFIDF_PCA, DFIDF_PCA = self.compute_tfidf_dfidf(BOW_PCA, tf_method = 'max')

        DOC_PCA = TFIDF_PCA.stack().reset_index().rename(columns = {0: 'TFIDF'})
        DOC_PCA = pd.merge(DOC_PCA.reset_index(), LIB.reset_index(), on = OHCO)\
            .set_index(OHCO).drop(columns = 'index')

        DOC_SUM = DOC_PCA.reset_index().groupby(OHCO).first()

        return DOC_SUM, TFIDF_PCA


    def compute_pca(self, X, DOC_SUM, k, norm_docs, center_by_mean, center_by_variance, doc_label):
        '''
        PURPOSE: compute the PCA from a given document-term count matrix

        INPUTS:
        X - The input matrix (dataframe)
        k - The number of components to generate
        norm_docs - Boolean flag to normalize the input matrix
        center_by_mean - Boolean flag to center the input matrix by its mean
        center_by_variance - Boolean flag to center the input matrix by its variance

        OUTPUTS:
        LOADINGS - The term-component matrix (dataframe)
        DCM - The document-component matrix (dataframe)
        COMPS - The component information table (dataframe)
        '''
        
        if norm_docs:
            X = (X.T / norm(X, 2, axis=1)).T

        if center_by_mean:
            X = X - X.mean()

        if center_by_variance:
            X = X / X.std()

        # Compute Covariance Matrix
        COV = X.T.dot(X) / (X.shape[0] - 1)

        # Decompose the Matrix
        eig_vals, eig_vecs = eigh(COV)

        # Convert eigen to dataframes
        EIG_VEC = pd.DataFrame(eig_vecs, index=COV.index, columns=COV.index)
        EIG_VAL = pd.DataFrame(eig_vals, index=COV.index, columns=['eig_val'])
        EIG_VAL.index.name = 'term_str'


        # Select principal components
        # Combine eigenvalues and eigenvectors
        EIG_PAIRS = EIG_VAL.join(EIG_VEC.T)

        # Compute explained variance
        EIG_PAIRS['exp_var'] = np.round((EIG_PAIRS.eig_val / EIG_PAIRS.eig_val.sum()) * 100, 2)

        # Pick top K components
        COMPS = EIG_PAIRS.sort_values('exp_var', ascending=False).head(k).reset_index(drop=True)
        COMPS.index.name = 'comp_id'
        COMPS.index = ["PC{}".format(i) for i in COMPS.index.tolist()]
        COMPS.index.name = 'pc_id'

        # See Projected Components onto Vocabulary (Loadings)
        LOADINGS = COMPS[COV.index].T
        LOADINGS.index.name = 'term_str'

        # Project Docs onto Components
        DCM = X.dot(COMPS[COV.index].T) 
        
        DCM_DOC = pd.merge(DCM, DOC_SUM, left_index = True, right_index = True)
        DCM_DOC['doc'] = DCM_DOC[doc_label]

        return LOADINGS, DCM_DOC, COMPS

    def vis_pcs(self, M, a, b, label='series_name', hover_name='doc', symbol=None, size=None):
        '''
        Purpose: visualize principal components for PCA
        '''
        
        fig = px.scatter(M, f"PC{a}", f"PC{b}", color=label, hover_name=hover_name, 
                         symbol=symbol, size=size,
                         marginal_x='box', height=800)
        fig.show()

    def vis_loadings(self, M, VOCAB, a=0, b=1, hover_name='term_str'):
        '''
        Purpose: visualize the loadings for PCA
        '''
        
        X = M.join(VOCAB)
        return px.scatter(X.reset_index(), f"PC{a}", f"PC{b}", 
                          text='term_str', size='i', color='max_pos', 
                          marginal_x='box', height=800)
    
    # TOPIC MODELS
    
    def generate_topic_model(self, BAG, TOKENS, ngram_range, n_terms, n_topics, max_iter, n_top_terms, tokens_filter=['NN', 'NNS'], max_df=1.0, min_df=1):
        '''
        Purpose: Generate the THETA, PHI, and TOPICS tables for topic modeling
        '''
        
        # filter for nouns
        DOCS = TOKENS[TOKENS.pos.isin(tokens_filter)]\
            .groupby(BAG).term_str\
            .apply(lambda x: ' '.join(x))\
            .to_frame()\
            .rename(columns={'term_str':'doc_str'})

        # create vector space

        count_engine = CountVectorizer(max_features=n_terms, ngram_range=ngram_range, stop_words='english', max_df=max_df, min_df=min_df)
        count_model = count_engine.fit_transform(DOCS.doc_str)
        TERMS = count_engine.get_feature_names_out()

        VOCAB = pd.DataFrame(index=TERMS)
        VOCAB.index.name = 'term_str'

        DTM = pd.DataFrame(count_model.toarray(), index=DOCS.index, columns=TERMS)

        VOCAB['doc_count'] = DTM.astype('bool').astype('int').sum()
        DOCS['term_count'] = DTM.sum(1)

        # generate model

        lda_engine = LDA(n_components=n_topics, max_iter=max_iter, learning_offset=50., random_state=0)

        TNAMES = [f"T{str(x).zfill(len(str(n_topics)))}" for x in range(n_topics)]

        # THETA

        lda_model = lda_engine.fit_transform(count_model)

        THETA = pd.DataFrame(lda_model, index=DOCS.index)
        THETA.columns.name = 'topic_id'
        THETA.columns = TNAMES

        # PHI

        PHI = pd.DataFrame(lda_engine.components_, columns=TERMS, index=TNAMES)
        PHI.index.name = 'topic_id'
        PHI.columns.name  = 'term_str'

        # TOPICS

        TOPICS = PHI.stack().to_frame('topic_weight').groupby('topic_id')\
            .apply(lambda x: x.sort_values('topic_weight', ascending=False)\
            .head(n_top_terms).reset_index().drop('topic_id', axis=1)['term_str'])

        TOPICS['label'] = TOPICS.apply(lambda x: x.name + ' ' + ', '.join(x[:n_top_terms]), 1)

        # sort topics by doc weight

        TOPICS['doc_weight_sum'] = THETA.sum()
        TOPICS['term_freq'] = PHI.sum(1) / PHI.sum(1).sum()

        return THETA, PHI, TOPICS
    
    # WORD EMBEDDINGS
    
    def generate_word_embeddings(self, CORPUS, OHCO, w2v_params):
        '''
        Purpose: Generate the coordinates and model for word2vec
        '''
        
        # Extract VOCAB
        VOCAB = CORPUS.term_str.value_counts().to_frame('n').sort_index()
        VOCAB.index.name = 'term_str'

        # Get the most frequently associated part-of-space category for each word.
        VOCAB['max_pos'] = CORPUS[['term_str','pos']].value_counts().unstack(fill_value=0).idxmax(1)
        VOCAB['pos_group'] = VOCAB['max_pos'].apply(lambda x: x[:2])
        
        # Get DOCS
        DOCS = CORPUS\
            .groupby(OHCO)\
            .term_str.apply(lambda  x:  x.tolist())\
            .reset_index()['term_str'].tolist()
        DOCS = [doc for doc in DOCS if len(doc) > 1] # Lose single word docs
        
        # Create Word2Vec model
        model = word2vec.Word2Vec(DOCS, **w2v_params)
        
        # Get coordinates
        coords = pd.DataFrame(
            dict(
                vector = [model.wv.get_vector(w) for w in model.wv.key_to_index], 
                term_str = model.wv.key_to_index.keys()
            )).set_index('term_str')
        
        # Create TSNE engine
        tsne_engine = TSNE(
            learning_rate=200,
            perplexity=20, 
            n_components=2, 
            init='random', 
            n_iter=1000, 
            random_state=42
        )
        vectors = np.array(coords.vector.to_list())
        tsne_model = tsne_engine.fit_transform(vectors)
        
        # Update coordinates
        coords['x'] = tsne_model[:,0]
        coords['y'] = tsne_model[:,1]

        if coords.shape[1] == 3:
            coords = coords.merge(VOCAB.reset_index(), on='term_str')
            coords = coords.set_index('term_str')
        
        return coords, model
    
    def complete_analogy(self, model, A, B, C, n=2):
        '''
        Purpose: Compute analogies for values in the word2vec
        '''
        
        try:
            cols = ['term', 'sim']
            return pd.DataFrame(model.wv.most_similar(positive=[B, C], negative=[A])[0:n], columns=cols)
        except KeyError as e:
            print('Error:', e)
            return None

    def get_most_similar(self, model, positive, negative=None):
        '''
        Purpose: Get similar words using the word2vec
        '''
        
        return pd.DataFrame(model.wv.most_similar(positive, negative), columns=['term', 'sim'])
    
    def plot_word_embeddings(self, coordinates):
        '''
        Purpose: Visualize the word embeddings
        '''
        
        fig = px.scatter(coordinates.reset_index(), 'x', 'y', 
           text='term_str', 
           color='pos_group', 
           hover_name='term_str',          
           size='n',
           height=1000).update_traces(
                mode='markers+text', 
                textfont=dict(color='black', size=14, family='Arial'),
                textposition='top center')
        
        fig.show()
    
    # SENTIMENT ANALYSIS
    
    def generate_sentiments(self, VOCAB, BOW, SALEX, OHCO):
        '''
        Purpose: Generate the emotion values given an OHCO level
        '''
        
        V = pd.concat([VOCAB, SALEX], join='inner', axis=1)
        
        emo_cols = "anger anticipation disgust fear joy sadness surprise trust sentiment".split()
        
        B = BOW.join(V[['max_pos'] + emo_cols], on='term_str', rsuffix='_v').dropna()
        
        for col in emo_cols:
            B[col] = B[col] * B.tfidf
        
        EMO = B.groupby(OHCO)[emo_cols].mean()
        
        EMO_thin = EMO.stack().to_frame().reset_index().rename(columns={0:'value', 'level_{}'.format(len(OHCO)): 'emo'})
        
        return EMO, EMO_thin, B
    
    def plot_sentiments(self, df, emo='sentiment'):
        '''
        Purpose: Plot the sentiment analysis values
        '''
        
        FIG = dict(figsize=(25, 5), legend=True, fontsize=14, rot=45)
        df[emo].plot(**FIG)
    
    def plot_thin_sentiments(self, df):
        '''
        Purpose: Plot the sentiment analysis values
        '''
        
        fig = px.line(df, x='season_id', y='value', color='emo')
        fig.show()
    
    def sample_sentences(self, SA_DF, OHCO):
        '''
        Purpose: Sample documents with the sentiments highlighted
        '''
        
        emo_cols = "anger anticipation disgust fear joy sadness surprise trust sentiment".split()
        
        SA_OHCO = SA_DF.groupby(OHCO)[emo_cols].mean()
        
        SA_OHCO['line_str'] = SA_DF.groupby(OHCO).term_str.apply(lambda x: x.str.cat(sep=' '))
        SA_OHCO['html_str'] = SA_DF.groupby(OHCO).html.apply(lambda x: x.str.cat(sep=' '))
        
        rows = []
        for idx in SA_OHCO.sample(20).index:

            valence = round(SA_OHCO.loc[idx, emo], 4)     
            t = 0
            if valence > t: color = '#ccffcc'
            elif valence < t: color = '#ffcccc'
            else: color = '#f2f2f2'
            z=0
            rows.append("""<tr style="background-color:{0};padding:.5rem 1rem;font-size:110%;">
            <td>{1}</td><td>{3}</td><td width="400" style="text-align:left;">{2}</td>
            </tr>""".format(color, valence, SA_OHCO.loc[idx, 'html_str'], idx))

        display(HTML('<style>#sample1 td{font-size:120%;vertical-align:top;} .sent-1{color:red;font-weight:bold;} .sent1{color:green;font-weight:bold;}</style>'))
        display(HTML('<table id="sample1"><tr><th>Sentiment</th><th>ID</th><th width="600">Sentence</th></tr>'+''.join(rows)+'</table>'))