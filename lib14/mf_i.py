import numpy as np
import ctypes
import os
import sys
import pandas as pd
import gzip
import json
import shutil
import numpy as np 
import pandas as pd 
import os
import re
import math
import time
import nltk
import joblib
import scipy.sparse
import datetime
from libmf import mf
from scipy. sparse import csr_matrix
from scipy. sparse import coo_matrix
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
import warnings; warnings.simplefilter('ignore')
#%matplotlib auto


if "LIBMF_OBJ" in os.environ:
    print("Using compiled .so file specified in LIBMF_OBJ:")
    compiled_src = os.environ["LIBMF_OBJ"]
else:
    site_pkgs = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    print("Using file found in {}:".format(site_pkgs))
    possible_objs = os.listdir(site_pkgs)
    filtered = [f for f in possible_objs if f[-3:] == '.so' and 'libmf' in f]
    if len(filtered) > 0:
        compiled_src = os.path.join(site_pkgs, filtered[0])
    else:
        raise IOError("Compiled .so file not found. If you know where it is, " 
		      "specify the path in the LIBMF_OBJ environment variable")
 
print(compiled_src)
mf = ctypes.CDLL(compiled_src)
c_float_p = ctypes.POINTER(ctypes.c_float)

''' libmf enums '''

P_L2_MFR = 0
P_L1_MFR = 1
P_KL_MFR = 2
P_LR_MFC = 5
P_L2_MFC = 6
P_L1_MFC = 7
P_ROW_BPR_MFOC = 10
P_COL_BPR_MFOC = 11

RMSE = 0
MAE = 1
GKL = 2
LOGLOSS = 5
ACC = 6
ROW_MPR = 10
COL_MPR = 11
ROW_AUC = 12
COL_AUC = 13

''' libmf enums '''


def get_default_options():
    options = [
        ("fun", ctypes.c_int, P_L2_MFR),
        ("k", ctypes.c_int, 8),
        ("nr_threads", ctypes.c_int, 12),
        ("nr_bins", ctypes.c_int, 26),
        ("nr_iters", ctypes.c_int, 20),
        ("lambda_p1", ctypes.c_float, 0.04),
        ("lambda_p2", ctypes.c_float, 0.0),
        ("lambda_q1", ctypes.c_float, 0.04),
        ("lambda_q2", ctypes.c_float, 0.0),
        ("eta", ctypes.c_float, 0.1),
        ("do_nmf", ctypes.c_bool, False),
        ("quiet", ctypes.c_bool, False),
        ("copy_data", ctypes.c_bool, True)
    ]
    return options


class MFModel(ctypes.Structure):
    _fields_ = [("fun", ctypes.c_int),
                ("m", ctypes.c_int),
                ("n", ctypes.c_int),
                ("k", ctypes.c_int),
                ("b", ctypes.c_float),
                ("P", c_float_p),
                ("Q", c_float_p)]


class MFParam(ctypes.Structure):
    _fields_ = [(o[0], o[1]) for o in get_default_options()]

options_ptr = ctypes.POINTER(MFParam)


class MF(object):
	
    def uhat(self):
        def parse(path):
          g = gzip.open(path, 'rb')
          for l in g:
            yield json.loads(l)

        def getDF(path):
          i = 0
          df = {}
          for d in parse(path):
            df[i] = d
            i += 1
          return pd.DataFrame.from_dict(df, orient='index')

        #with open('Appliances_5_test', 'rb') as f_in, gzip.open('test.json.gz', 'wb') as f_out:
          #shutil.copyfileobj(f_in, f_out)
        df = getDF('Appliances_5.json.gz') #讀檔

        df.insert(df.shape[1], 'word', 0) # 增加一欄位用以儲存評論之名詞

        userNo = len(np.unique(df.reviewerID)) #使用者總數
        itemNo = len(np.unique(df.asin)) #商品總數
        D = 29 #隱含特徵數
        T = 36 #時間點數
        merge = 2 #每幾個時間點合併 
        user_list = np.unique(df.reviewerID) #取出所有不同使用者
        item_list = np.unique(df.asin) #取出所有不同商品
        rating = np.zeros((itemNo, userNo)) #創建矩陣
        user_list = pd.Series(user_list.tolist())
        item_list = pd.Series(item_list.tolist())

        df.reviewText = df.reviewText.astype(str)

        bow = []
        stop_words = set(stopwords.words('english')) #停用詞
        for i in range(len(df)): #建立包含所有評論內名詞的詞袋

            text= df.reviewText[i].lower()
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = nltk.word_tokenize(text)
            tags = nltk.pos_tag(tokens)
            lemmatizer = WordNetLemmatizer()
            nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')] #留下名詞
            nouns = [w for w in nouns if not w.lower() in stop_words] #去除停用詞
            nouns = [w for w in nouns if w.isalpha()] #去除標點符號
            nouns = [lemmatizer.lemmatize(w) for w in nouns] #詞幹化
            df.word[i] = nouns
            bow += nouns
        bow = np.unique(bow) #留下不重複的

        i_tfidf = pd.DataFrame() #算I矩陣(TF-IDF)
        for i in range(len(df)):
            vectorizer = TfidfVectorizer(stop_words='english', vocabulary = bow)
            tfidf = vectorizer.fit_transform([df.reviewText[i]])
            result = pd.DataFrame(tfidf.toarray(), columns=vectorizer.vocabulary_)
            i_tfidf = i_tfidf.append(result, ignore_index=True)
        i_tfidf = np.transpose(i_tfidf)

        model = NMF(n_components = D, random_state = 5)
        model.fit(i_tfidf)
        nmf_features = model.transform(i_tfidf)
        h = nmf_features
        w = model.components_

        temp = pd.DataFrame(columns = ['asin', 'userID', 'review']) #把同一物品/使用者的評論單詞放在一起變新的評論
        temp.asin = df.asin
        temp.review = df.word
        temp.userID = df.reviewerID
        psiJ = temp.groupby(by = 'asin')['review'].sum() #同物品
        space = ' '
        for i  in range(len(psiJ)):
            psiJ[i] = space.join(psiJ[i])

        psiI = temp.groupby(by = 'userID')['review'].sum() #同使用者
        space = ' '
        for i  in range(len(psiI)):
            psiI[i] = space.join(psiI[i])

        f_item = pd.DataFrame() #算psi的TF-IDF
        for i in range(len(psiJ)):
            vectorizer = TfidfVectorizer(stop_words='english', vocabulary = bow)
            tfidf = vectorizer.fit_transform([psiJ[i]])
            result = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
            f_item = f_item.append(result, ignore_index=True)


        f_user = pd.DataFrame() #算psi的TF-IDF
        for i in range(len(psiI)):
            vectorizer = TfidfVectorizer(stop_words='english', vocabulary = bow)
            tfidf = vectorizer.fit_transform([psiI[i]])
            result = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
            f_user = f_user.append(result, ignore_index=True)

        v_hat = np.dot(f_item, h) #投射到U_hat和V_hat
        u_hat = np.dot(f_user, h)
        return u_hat

    def vhat(self):
        def parse(path):
          g = gzip.open(path, 'rb')
          for l in g:
            yield json.loads(l)

        def getDF(path):
          i = 0
          df = {}
          for d in parse(path):
            df[i] = d
            i += 1
          return pd.DataFrame.from_dict(df, orient='index')

        #with open('Appliances_5_test', 'rb') as f_in, gzip.open('test.json.gz', 'wb') as f_out:
          #shutil.copyfileobj(f_in, f_out)
        df = getDF('Appliances_5.json.gz') #讀檔

        df.insert(df.shape[1], 'word', 0) # 增加一欄位用以儲存評論之名詞

        userNo = len(np.unique(df.reviewerID)) #使用者總數
        itemNo = len(np.unique(df.asin)) #商品總數
        D = 29 #隱含特徵數
        T = 36 #時間點數
        merge = 2 #每幾個時間點合併 
        user_list = np.unique(df.reviewerID) #取出所有不同使用者
        item_list = np.unique(df.asin) #取出所有不同商品
        rating = np.zeros((itemNo, userNo)) #創建矩陣
        user_list = pd.Series(user_list.tolist())
        item_list = pd.Series(item_list.tolist())

        df.reviewText = df.reviewText.astype(str)

        bow = []
        stop_words = set(stopwords.words('english')) #停用詞
        for i in range(len(df)): #建立包含所有評論內名詞的詞袋

            text= df.reviewText[i].lower()
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = nltk.word_tokenize(text)
            tags = nltk.pos_tag(tokens)
            lemmatizer = WordNetLemmatizer()
            nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')] #留下名詞
            nouns = [w for w in nouns if not w.lower() in stop_words] #去除停用詞
            nouns = [w for w in nouns if w.isalpha()] #去除標點符號
            nouns = [lemmatizer.lemmatize(w) for w in nouns] #詞幹化
            df.word[i] = nouns
            bow += nouns
        bow = np.unique(bow) #留下不重複的

        i_tfidf = pd.DataFrame() #算I矩陣(TF-IDF)
        for i in range(len(df)):
            vectorizer = TfidfVectorizer(stop_words='english', vocabulary = bow)
            tfidf = vectorizer.fit_transform([df.reviewText[i]])
            result = pd.DataFrame(tfidf.toarray(), columns=vectorizer.vocabulary_)
            i_tfidf = i_tfidf.append(result, ignore_index=True)
        i_tfidf = np.transpose(i_tfidf)

        model = NMF(n_components = D, random_state = 5)
        model.fit(i_tfidf)
        nmf_features = model.transform(i_tfidf)
        h = nmf_features
        w = model.components_

        temp = pd.DataFrame(columns = ['asin', 'userID', 'review']) #把同一物品/使用者的評論單詞放在一起變新的評論
        temp.asin = df.asin
        temp.review = df.word
        temp.userID = df.reviewerID
        psiJ = temp.groupby(by = 'asin')['review'].sum() #同物品
        space = ' '
        for i  in range(len(psiJ)):
            psiJ[i] = space.join(psiJ[i])

        psiI = temp.groupby(by = 'userID')['review'].sum() #同使用者
        space = ' '
        for i  in range(len(psiI)):
            psiI[i] = space.join(psiI[i])

        f_item = pd.DataFrame() #算psi的TF-IDF
        for i in range(len(psiJ)):
            vectorizer = TfidfVectorizer(stop_words='english', vocabulary = bow)
            tfidf = vectorizer.fit_transform([psiJ[i]])
            result = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
            f_item = f_item.append(result, ignore_index=True)


        f_user = pd.DataFrame() #算psi的TF-IDF
        for i in range(len(psiI)):
            vectorizer = TfidfVectorizer(stop_words='english', vocabulary = bow)
            tfidf = vectorizer.fit_transform([psiI[i]])
            result = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
            f_user = f_user.append(result, ignore_index=True)

        v_hat = np.dot(f_item, h) #投射到U_hat和V_hat
        u_hat = np.dot(f_user, h)
        return v_hat
	
    def __init__(self, *args, **kwargs):
        self.model = None
        self._options = MFParam()
        self.i = None
        self.j = None
        for kw in kwargs:
            if kw not in [i[0] for i in get_default_options()]:
                print("Unrecognized keyword argument '{0}={1}'".format(kw, kwargs[kw]))

        for item in get_default_options():
            if item[0] not in kwargs:
                value = item[2]
            else:
                value = kwargs[item[0]]

            if item[0] is "fun":
                self._options.fun = ctypes.c_int(value)
            elif item[0] is "k":
                self._options.k = ctypes.c_int(value)
            elif item[0] is "nr_threads":
                self._options.nr_threads = ctypes.c_int(value)
            elif item[0] is "nr_bins":
                self._options.nr_bins = ctypes.c_int(value)
            elif item[0] is "nr_iters":
                self._options.nr_iters = ctypes.c_int(value)
            elif item[0] is "lambda_p1":
                self._options.lambda_p1 = ctypes.c_float(value)
            elif item[0] is "lambda_p2":
                self._options.lambda_p2 = ctypes.c_float(value)
            elif item[0] is "lambda_q1":
                self._options.lambda_q1 = ctypes.c_float(value)
            elif item[0] is "lambda_q2":
                self._options.lambda_q2 = ctypes.c_float(value)
            elif item[0] is "eta":
                self._options.eta = ctypes.c_float(value)
            elif item[0] is "do_nmf":
                self._options.do_nmf = ctypes.c_bool(value)
            elif item[0] is "quiet":
                self._options.quiet = ctypes.c_bool(value)
            elif item[0] is "copy_data":
                self._options.copy_data = ctypes.c_bool(value)

    def predict(self, X):
        """
        assuming we have already run the fit method, predict the values at certain indices of the data matrix
        :param X: (n, 2) shaped numpy array
        :return: numpy array of length n
        """
        if self.model is None:
            return LookupError("no model data is saved, try running model.mf_fit(...) first")
        ensure_width(X, 2)
        nnx = X.shape[0]
        out = np.zeros(nnx)
        out = out.astype(np.float32)
        X = X.astype(np.float32)
        X_p = X.ctypes.data_as(c_float_p)
        nnx_p = ctypes.c_int(nnx)
        mf.pred_model_interface(nnx_p, X_p, ctypes.c_void_p(out.ctypes.data), ctypes.byref(self.model))
        return out

    def fit(self, X):
        """
        factorize the i x j data matrix X into (j, k) (k, i) sized matrices stored in MF.model
        :param X: (n, 3) shaped numpy array [known index and values of the data matrix]
        """
        ensure_width(X, 3)
        d = X.astype(np.float32)
        data_p = d.ctypes.data_as(c_float_p)
        nnx = ctypes.c_int(X.shape[0])
        mf.fit_interface.restype = ctypes.POINTER(MFModel)
        mf.fit_interface.argtypes = (ctypes.c_int, c_float_p, options_ptr)
        out = mf.fit_interface(nnx, data_p, self._options)
        self.model = out.contents

    def mf_cross_validation(self, X, folds=5):
        """
        :param X: (n, 3)
        :param folds: number of train / test splits
        :return: average score across all folds
        """
        ensure_width(X, 3)
        d = X.astype(np.float32)
        data_p = d.ctypes.data_as(c_float_p)
        nnx = ctypes.c_int(X.shape[0])
        mf.cross_valid_interface.restype = ctypes.c_double
        mf.cross_valid_interface.argtypes = (ctypes.c_int, c_float_p, options_ptr, ctypes.c_int)
        score = mf.cross_valid_interface(nnx, data_p, self._options, folds)
        return score

    def mf_train_test(self, X, V):
        ensure_width(X, 3)
        ensure_width(V, 3)
        nnx = ctypes.c_int(X.shape[0])
        nnx_valid = ctypes.c_int(V.shape[0])

        train_p = X.astype(np.float32)
        train_p = train_p.ctypes.data_as(c_float_p)

        test_p = V.astype(np.float32)
        test_p = test_p.ctypes.data_as(c_float_p)

        mf.train_valid_interface.restype = ctypes.POINTER(MFModel)
        mf.train_valid_interface.argtypes = (ctypes.c_int, ctypes.c_int, c_float_p, c_float_p, options_ptr)
        out = mf.train_valid_interface(nnx, nnx_valid, train_p, test_p, self._options)
        self.model = out.contents

    def q_factors(self):
        if self.model is None:
            return LookupError("no model data is saved, try running model.mf_fit(...) first")
        tmp = uhat()
        tmp.reshape(self.model.n * self.model.k)
        out = uhat
        out = out.astype(np.float32)
        mf.get_Q(ctypes.c_void_p(out.ctypes.data), ctypes.byref(self.model))
        return out.reshape((self.model.n, self.model.k))

    def p_factors(self):
        if self.model is None:
            return LookupError("no model data is saved, try running model.mf_fit(...) first")
        tmp = vhat()
	tmp.reshape(self.model.m * self.model.k)
        out = vhat
        out = out.astype(np.float32)
        mf.get_P(ctypes.c_void_p(out.ctypes.data), ctypes.byref(self.model))
        return out.reshape((self.model.m, self.model.k))


def ensure_width(x, width):
    if x.shape[1] != width:
        raise ValueError("must be sparse array of shape (n, {0})", width)


def generate_test_data(xs, ys, k, indices_only=False):
    rx = np.random.random_integers(0, xs, k)
    ry = np.random.random_integers(0, ys, k)
    rv = np.random.rand(k)
    return np.vstack((rx, ry, rv)).transpose().copy() if not indices_only else np.vstack((rx,ry)).transpose().copy()

