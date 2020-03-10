# This code is referred to https://github.com/xuqiantong/GAN-Metrics
# refactorizing above souece code with numpy & keras version
import os
import shutil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import ot
import torch

from scipy import linalg

from tqdm import tqdm

import tensorflow as tf


# keras module
from keras.models import Model, Sequential
from keras.layers import *
from keras.applications import DenseNet121
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D

# private module
from data_load_utils import loadImgFileFromDir

def distance(X, Y, sqrt): # X : (N, )
    nX = X.size(0)
    nY = Y.size(0)
    # nX = X.shape[0]
    # nY = Y.shape[0]
    #rint("nX", nX, "nY", nY) # nX 100 nY 100

    #print("X.size() 1 ", X.size()) # [100, 3, 269, 224]
    X = X.view(nX,-1) # flatten
    # print("X.size() after view(nX, -1)", X.shape) # [100, 180768]
    X2 = (X*X).sum(1).resize_(nX,1)
    #print("X2.size()", X2.size()) # [100, 1]
    Y = Y.view(nY,-1)
    #print("Y.size()", Y.size()) # [100, 180768]
    Y2 = (Y*Y).sum(1).resize_(nY,1)
    #print("Y2.size()", Y2.size()) # [100, 1]

    M = torch.zeros(nX, nY)
    #print("M", M.size()) # [100, 100]
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))
    #print("M copy_", M.size()) # [100, 100]
    # M = np.zeros((nX, nY))
    # M.copy(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
    #         2 * np.matmul(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.absolute()) / 2).sqrt()

    return M



class ConvNetFeatureSaver_Keras(object):
    def __init__(self, model='resnet34', batchSize=64, input_shape=(224, 224, 3), sampleSize=None):
        '''
        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
        '''

        self.model = model
        self.batch_size = batchSize
        self.input_shape = input_shape
        self.sampleSize = sampleSize

        if "resnet50" in self.model:
            self.resnet50 = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
            x = self.resnet50.output
            x = GlobalAveragePooling2D()(x)
            self.resnet50 = Model(inputs=self.resnet50.input, outputs=x)
        elif "densenet121" in self.model:
            self.densenet_model = DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape)
            x = self.densenet_model.output
            x = GlobalAveragePooling2D()(x)
            self.densenet_model = Model(inputs=self.densenet_model.input, outputs=x)

    def resize(self, img):
        return img.resize((self.input_shape[0], self.input_shape[1]))

    def feature_extractor(self, source_data, func=None):
        if isinstance(source_data, str):
            imgs = loadImgFileFromDir(source_data, func, self.sampleSize)
        elif isinstance(source_data, np.ndarray):
            imgs = source_data
        print('extracting features...')
        feature_pixl, feature_conv, feature_smax, feature_logit = [], [], [], []

        if "resnet50" in self.model:
            fconv = self.resnet50.predict(imgs['image'])
        elif "densenet121" in self.model:
            fconv = self.densenet_model.predict(imgs['image'])
        else:
            raise NotImplementedError
        feature_conv = fconv
        feature_conv = np.array(feature_conv)
        
        print("feature_conv", feature_conv.shape)
        
        
        return feature_conv

    def feature_extractor_from_npMat(self, source_data, func=None):

        imgs = source_data
        print("feature_extractor_from_npMat:",imgs.shape)
        print('extracting features...')
        feature_pixl, feature_conv, feature_smax, feature_logit = [], [], [], []

        
        if "resnet50" in self.model:
            fconv = self.resnet50.predict(imgs)
        elif "densenet121" in self.model:
            fconv = self.densenet_model.predict(imgs)
        else:
            raise NotImplementedError

        #feature_conv.append(fconv)
        feature_conv = fconv
        feature_conv = np.array(feature_conv)
        print("feature_conv", feature_conv.shape)
       
        return feature_conv


def giveName(iter):  # 7 digit name.
    ans = str(iter)
    return ans.zfill(7)

eps = 1e-20

def inception_score(X):
    # print("inception_score() : X", X.size()) # torch.Size([1024]) # [100, 1000]
    kl = X * ((X + eps).log() - (X.mean(0) + eps).log().expand_as(X))
    # print("kl", kl)
    # print("kl", kl.size()) # kl torch.Size([1024])
    score = np.exp(kl.sum(1).mean()) # i think here this is weird

    return score

def mode_score(X, Y):
    kl1 = X * ((X + eps).log() - (X.mean(0) + eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0) + eps).log() - (Y.mean(0) + eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())

    return score

def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
            np.trace(C + C_w - 2 * C_C_w_sqrt)
    return np.sqrt(score)

#import ot
def wasserstein(M, sqrt):
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([], [], M.numpy())
    return emd
    #return None

import math
def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean() # a mean value
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma)) # ?? -> torch.exp(-(Mxx-scale)**2 / (2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

    return mmd

class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

def knn(Mxx, Mxy, Myy, k, sqrt):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))
                ).topk(k, 0, False)

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

    s = Score_knn()
    s.tp = (pred * label).sum()
    s.fp = (pred * (1 - label)).sum()
    s.fn = ((1 - pred) * label).sum()
    s.tn = ((1 - pred) * (1 - label)).sum()
    s.precision = s.tp / (s.tp + s.fp + 1e-10)
    s.recall = s.tp / (s.tp + s.fn + 1e-10)
    s.acc_real = s.tp / (s.tp + s.fn)
    s.acc_fake = s.tn / (s.tn + s.fp)
    s.acc = torch.eq(label, pred).float().mean()
    s.k = k

    return s



class Score:
    emd = 0
    mmd = 0
    knn = None

def compute_score(real, fake, k=1, sigma=1, sqrt=True):

    Mxx = distance(real, real, False)
    Mxy = distance(real, fake, False)
    Myy = distance(fake, fake, False)

    s = Score()
    s.emd = wasserstein(Mxy, sqrt)
    s.mmd = mmd(Mxx, Mxy, Myy, sigma)
    s.knn = knn(Mxx, Mxy, Myy, k, sqrt)

    return s


def compute_score_raw_from_PNG(imageSize, sampleSize, batchSize,
                  saveFolder_r, saveFolder_f, dataroot_r, dataroot_f, conv_model='resnet50', workers=4, only_feature_space=True):
    if dataroot_r is not None :
        sampleData(dataroot_r, imageSize, sampleSize, batchSize, saveFolder_r)
    if dataroot_f is not None :
        sampleData(dataroot_f, imageSize, sampleSize, batchSize, saveFolder_f)
    # convnet_feature_saver = ConvNetFeatureSaver(model=conv_model, batchSize=batchSize, workers=workers)
    # feature_r = convnet_feature_saver.save(saveFolder_r)
    # feature_f = convnet_feature_saver.save(saveFolder_f)

    if conv_model=="densenet121":
        input_shape = (224, 224, 3)
    elif conv_model=="resnet50":
        input_shape = (197, 197, 3)
    convnet_feature_saver = ConvNetFeatureSaver_Keras(model=conv_model, input_shape=input_shape, sampleSize=sampleSize)
    #print("compute_score_raw save_Folder_r", saveFolder_r)
    def _resize(img):
        return img.resize((input_shape[0], input_shape[1])).convert("RGB")
    feature_r = convnet_feature_saver.feature_extractor(saveFolder_r, _resize)
    feature_f = convnet_feature_saver.feature_extractor(saveFolder_f, _resize)
    print("feature_r shape", feature_r.shape)
    print("feature_f shape", feature_f.shape)
    if only_feature_space is True:
        score = np.zeros(10)
        feature_r = torch.tensor(feature_r)
        feature_f = torch.tensor(feature_f)
        #print("compute_score_raw_from_PNG:", feature_f.size())
        # Mxx = distance(feature_r, feature_r, False)
        # Mxy = distance(feature_r, feature_f, False)
        # Myy = distance(feature_f, feature_f, False)
        Mxx = distance(feature_r, feature_r, False)
        Mxy = distance(feature_r, feature_f, False)
        Myy = distance(feature_f, feature_f, False)
        #score[0] = wasserstein(Mxy, True)
        score[1] = mmd(Mxx, Mxy, Myy, 1)
        tmp = knn(Mxx, Mxy, Myy, 1, False)
        score[2:7] = tmp.acc, tmp.acc_real, tmp.acc_fake, tmp.precision, tmp.recall
        score[7] = inception_score(feature_f)
        score[8] = mode_score(feature_r, feature_f)
        score[9] = fid(feature_r, feature_f)

        return score
    # 4 feature spaces and 7 scores + incep + modescore + fid
    score = np.zeros(4 * 7 + 3)
    for i in range(0, 4):
        print('compute score in space: ' + str(i))
        Mxx = distance(feature_r[i], feature_r[i], False)
        Mxy = distance(feature_r[i], feature_f[i], False)
        Myy = distance(feature_f[i], feature_f[i], False)

        score[i * 7] = wasserstein(Mxy, True)
        score[i * 7 + 1] = mmd(Mxx, Mxy, Myy, 1)
        tmp = knn(Mxx, Mxy, Myy, 1, False)
        score[(i * 7 + 2):(i * 7 + 7)] = \
            tmp.acc, tmp.acc_real, tmp.acc_fake, tmp.precision, tmp.recall

        if i == 0:
            # feature_pixl
            print("feature_pixl")
            print("wasserstein in feature_conv", score[i * 7])
            print("mmd in feature_conv", score[i * 7 + 1])
            print("knn.acc in feature_conv", score[i * 7 + 2])
            print("knn.acc_real in feature_conv", score[i * 7 + 3])
            print("knn.acc_fake in feature_conv", score[i * 7 + 4])
            print("knn.precision in feature_conv", score[i * 7 + 5])
            print("knn.recall in feature_conv", score[i * 7 + 6])
        elif i == 1:
            # feature_conv
            print("feature_conv")
            print("wasserstein in feature_conv", score[i * 7])
            print("mmd in feature_conv", score[i * 7 + 1])
            print("knn.acc in feature_conv", score[i * 7 + 2])
            print("knn.acc_real in feature_conv", score[i * 7 + 3])
            print("knn.acc_fake in feature_conv", score[i * 7 + 4])
            print("knn.precision in feature_conv", score[i * 7 + 5])
            print("knn.recall in feature_conv", score[i * 7 + 6])
        elif i == 2:
            # feature_logit
            print("feature_logit")
            # print("wasserstein in feature_logit", score[0])
            # print("mmd in feature_logit", score[1])
            # print("knn.acc in feature_logit", score[2])
            # print("knn.acc_real in feature_logit", score[3])
            # print("knn.acc_fake in feature_logit", score[4])
            # print("knn.precision in feature_logit", score[5])
            # print("knn.recall in feature_logit", score[6])
        elif i == 3:
            # feature_smax
            print("feature_smax")
            # print("wasserstein in feature_smax", score[0])
            # print("mmd in feature_smax", score[1])
            # print("knn.acc in feature_smax", score[2])
            # print("knn.acc_real in feature_smax", score[3])
            # print("knn.acc_fake in feature_smax", score[4])
            # print("knn.precision in feature_smax", score[5])
            # print("knn.recall in feature_smax", score[6])
    score[28] = inception_score(feature_f[3])
    score[29] = mode_score(feature_r[3], feature_f[3])
    score[30] = fid(feature_r[3], feature_f[3])

    return score

# copy a set of slices to other directory
def extract_slice(target_dir, dest_dir, ind):
    save_dir = os.path.join(dest_dir, str(ind))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    childs = [os.path.join(target_dir, child) for child in os.listdir(target_dir)]

    for child in childs:
        basename = os.path.basename(child)
        onlyfilename = basename.split(".")[0]
        file_slice = onlyfilename.split("_")[1]
        if str(file_slice) == str(ind):
            source_file = os.path.join(target_dir, basename)
            dest_file = os.path.join(dest_dir, str(ind), basename)
            shutil.copy(source_file, dest_file)
    return

if '__main__' == __name__:
    print("hello world")
    imageSize = None  # size of image to resize
    sampleSize = 69  # Num to sample
    batchSize = 10
    workers = 1


    dataroot_r = None
    saveFolder_r = "..\\training\\bapl3"
    
    dataroot_f = None
    saveFolder_f = "..\\original\\bapl3"
    
    feature_conv_wasserstein = []
    feature_conv_mmd = [] # mmd
    feature_conv_knn_acc = []
    feature_conv_knn_acc_real = []
    feature_conv_knn_acc_fake = []
    feature_conv_knn_acc_precision = []
    feature_conv_knn_acc_recall = []

    feature_conv_inception_score = []
    feature_conv_mode_score = []
    feature_conv_fid = []

    #for ind in tqdm(range(100, 2900, 100)):
    for ind in tqdm(range(0, 36)):

        _saveFolder_r = saveFolder_r + "\\" + str(ind)
        _saveFolder_f = saveFolder_f + "\\" + str(ind)
        score = compute_score_raw_from_PNG(imageSize, sampleSize, batchSize,
                                           _saveFolder_r, _saveFolder_f, dataroot_r, dataroot_f,
                                           conv_model='densenet121', workers=4)
        i = 0
        feature_conv_wasserstein.append(score[i * 7])
        feature_conv_mmd.append(score[1])
        feature_conv_knn_acc.append(score[2])
        feature_conv_knn_acc_real.append(score[3])
        feature_conv_knn_acc_fake.append(score[4])
        print("feature_conv_mmd: ", score[1])
        print("feature_conv_knn_acc: ", score[2])
        print("feature_conv_knn_acc_real: ", score[3])
        print("feature_conv_knn_acc_fake: ", score[4])

        feature_conv_knn_acc_precision.append(score[i * 7 + 5])
        feature_conv_knn_acc_recall.append(score[i * 7 + 6])

        feature_conv_inception_score.append(score[7])
        feature_conv_mode_score.append(score[8])
        feature_conv_fid.append(score[9])

    result_dict = dict()

    result_dict["wasserstein"] = feature_conv_wasserstein
    result_dict["mmd"] = feature_conv_mmd
    result_dict["knn_acc"] = feature_conv_knn_acc
    result_dict["knn_acc_real"] = feature_conv_knn_acc_real
    result_dict["knn_acc_fake"] = feature_conv_knn_acc_fake
    result_dict["knn_precision"] = feature_conv_knn_acc_precision
    result_dict["knn_recall"] = feature_conv_knn_acc_recall

    result_dict["inception_score"] = feature_conv_inception_score
    result_dict["mode_score"] = feature_conv_mode_score
    result_dict["fid"] = feature_conv_fid

    result_df = pd.DataFrame(result_dict)
    result_filename = "..\\metrics\\training_original\\t_bapl3_g_bapl3\\training_vs_original_bapl3_3_gray_slice.xlsx"
    result_df.to_excel(result_filename)
