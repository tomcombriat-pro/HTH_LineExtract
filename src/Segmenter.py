import catboost as cb
from skimage.feature import multiscale_basic_features
from skimage.measure import label
import skimage.io as io
from skimage.filters import threshold_otsu
from functools import partial
import numpy as np
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt


def cf0(mat):
    return multiscale_basic_features(mat,intensity = False).reshape((len(mat)*len(mat[0]),-1))



class Segmenter:

    def __init__(self, clf=cb.CatBoostClassifier(),cf = cf0):
        self.clf = clf
        self.cf = cf
        self.threshold = 0.5

    def fit(self,training_features, training_labels, **kwargs):
        self.clf.fit(training_features,training_labels,**kwargs)
        

    def segment(self, img):
        pred = self.clf.predict_proba(self.cf(img)) > self.threshold
        if (img.ndim == 3):
            shape = (len(img),len(img[0]))
        else:
            shape = img.shape
        return pred[:,1].reshape(shape)

    def get_proba(self,img):
        pred = self.clf.predict_proba(self.cf(img))
        if (img.ndim == 3):
            shape = (len(img),len(img[0]))
        else:
            shape = img.shape
        return pred[:,1].reshape(shape)
        

    

def segment_fill(img, segmenter):
    pred = segmenter.segment(img)
    pred = binary_fill_holes(pred)
    return pred

