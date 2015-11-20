__author__ = 'jessyli'
import os
import timeit
import sys

import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import math
import operator
from scipy import sparse
from cvxopt import spmatrix
from numpy import linalg as LA
from scipy.stats.stats import pearsonr
from array import array
from sklearn.metrics.pairwise import cosine_distances
import csv

def readingtrain(filename):
    f = open(filename)
    # csv_f = csv.reader(f)
    row = []
    col = []
    data = []
    for lines in f.readlines():
        l = lines.split(",")
        row.append(int(l[0]))
        col.append(int(l[1]))
        if((int(l[2])-3)==0):
            data.append(1e-7)
        else:
            data.append(int(l[2])-3)
    rowsize = max(row)
    colsize = max(col)
    comatrix = coo_matrix((data, (row, col)), shape=(rowsize+1, colsize+1))

    return comatrix

def readingdevqueries(filename):
    f = open(filename)
    row = []
    col = []
    data = []
    for lines in f.readlines():
        l = lines.split()
        size = len(l)
        temprow = [int(l[0])]*(size-1)
        row.extend(temprow)
        tempcol = []
        tempdata = []
        for i in range(1,size):
            l1 = l[i].split(":")
            tempcol.append(int(l1[0]))
            tempdata.append(int(l1[1]))
        col.extend(tempcol)
        data.extend(tempdata)
    rowsize = max(row)
    colsize = max(col)
    matrixsize = max(rowsize, colsize)
    comatrix = coo_matrix((data, (row, col)), shape=(matrixsize+1, matrixsize+1))
    return comatrix
def readingdev(filename):
    f = open(filename)
    row = []
    col = []
    data = []
    count = 0
    for lines in f.readlines():
        l = lines.split(",")
        row.append(count)
        col.append(0)
        data.append(int(l[0]))
        row.append(count)
        col.append(1)
        data.append(int(l[1]))
        count = count+1
    rowsize = max(row)
    colsize = max(col)
    comatrix = coo_matrix((data, (row, col)), shape=(rowsize+1, colsize+1))
    return comatrix
def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
def RepresentsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def readbipartite():
    f1 = open("doc_centroid.txt")
    row = []
    col = []
    data = []
    for lines in f1.readlines():
        l = lines.split(" ")
        r = int(l[0])
        l1 = l[1].split(",")

        for i in range(len(l1)):
            row.append(r)

            l2 = l1[i].split(":")
            if RepresentsInt(l2[0])==False:
                a = l2[0][1:]
                col.append(int(a))
                if RepresentsFloat(l2[1])==False:
                    a = l2[1][:-2]
                    data.append(float(a))
                else:
                    data.append(float(l2[1]))
            elif RepresentsFloat(l2[1])==False:
                a = l2[1][:-2]
                col.append(int(l2[0]))
                data.append(float(a))
            else:
                col.append(int(l2[0]))
                data.append(float(l2[1]))
    rowsize = max(row)
    colsize = max(col)
    comatrixmovie = coo_matrix((data, (row, col)), shape=(rowsize+1, colsize+1))
    cosinedistancemovie = np.zeros(shape=(comatrixmovie.shape[0], comatrixmovie.shape[0]))
    dotdistancemovie = comatrixmovie.todense()*(comatrixmovie.transpose().todense())
    cosinedistancemovie1 = np.zeros(shape=(comatrixmovie.shape[0], comatrixmovie.shape[0]))
    vec = np.linalg.norm(comatrixmovie.todense(), axis=1)
    for i in range(0, comatrixmovie.shape[0]):
        cosinedistancemovie[i] = vec*vec[i]
        cosinedistancemovie1[i] = np.divide(dotdistancemovie[i], (cosinedistancemovie[i]))
    f2 = open("word_centroid.txt")
    row = []
    col = []
    data = []
    for lines in f2.readlines():
        l = lines.split(" ")
        r = int(l[0])
        l1 = l[1].split(",")
        for i in range(len(l1)):
            col.append(r)
            l2 = l1[i].split(":")
            if RepresentsInt(l2[0])==False:
                a = l2[0][1:]
                row.append(int(a))
                if RepresentsFloat(l2[1])==False:
                    a = l2[1][:-2]
                    data.append(float(a))
                else:
                    data.append(float(l2[1]))
            elif RepresentsFloat(l2[1])==False:
                a = l2[1][:-2]
                row.append(int(l2[0]))
                data.append(float(a))
            else:
                row.append(int(l2[0]))
                data.append(float(l2[1]))
    rowsize = max(row)
    colsize = max(col)
    comatrixuser = coo_matrix((data, (row, col)), shape=(rowsize+1, colsize+1))
    cosinedistanceuser = np.zeros(shape=(comatrixuser.shape[1], comatrixuser.shape[1]))
    dotdistanceuser = comatrixuser.transpose().todense()*(comatrixuser.todense())
    cosinedistanceuser1 = np.zeros(shape=(comatrixuser.shape[1], comatrixuser.shape[1]))
    # pccdistanceuser = np.zeros(shape=(comatrix.shape[1], comatrix.shape[1]))
    vec = np.linalg.norm(comatrixuser.todense(), axis=0)
    for i in range(0, comatrixuser.shape[1]):
        cosinedistanceuser[i] = vec*vec[i]
        cosinedistanceuser1[i] = np.divide(dotdistanceuser[i], (cosinedistanceuser[i]))
    return (cosinedistancemovie, dotdistancemovie, cosinedistanceuser, dotdistanceuser)
def readcluster():
    f = open("word_cluster.txt")
    usercluster = dict()
    for lines in f.readlines():
        l =lines.split(" ")
        usercluster[int(l[0])]=(int(l[1]))
    f1 = open("doc_cluster.txt")
    moviecluster = dict()
    for lines in f1.readlines():
        l =lines.split(" ")
        moviecluster[int(l[0])]=(int(l[1]))
    return (usercluster, moviecluster)

def user():
    script_dir = os.path.dirname(__file__)
    rel_path = "HW4_data/train.csv"
    abs_file_path = os.path.join(script_dir, rel_path)
    comatrix = readingtrain(abs_file_path)
    cosinedistanceuser = np.zeros(shape=(comatrix.shape[1], comatrix.shape[1]))
    dotdistanceuser = comatrix.transpose().todense()*(comatrix.todense())
    cosinedistanceuser1 = np.zeros(shape=(comatrix.shape[1], comatrix.shape[1]))
    # pccdistanceuser = np.zeros(shape=(comatrix.shape[1], comatrix.shape[1]))
    vec = np.linalg.norm(comatrix.todense(), axis=0)
    for i in range(0, comatrix.shape[1]):
        cosinedistanceuser[i] = vec*vec[i]
        cosinedistanceuser1[i] = np.divide(dotdistanceuser[i], (cosinedistanceuser[i]))
    return (dotdistanceuser, cosinedistanceuser1)

def movie():
    script_dir = os.path.dirname(__file__)
    rel_path = "HW4_data/train.csv"
    abs_file_path = os.path.join(script_dir, rel_path)
    comatrix = readingtrain(abs_file_path)
    cosinedistancemovie = np.zeros(shape=(comatrix.shape[0], comatrix.shape[0]))
    dotdistancemovie = comatrix.todense()*(comatrix.transpose().todense())
    cosinedistancemovie1 = np.zeros(shape=(comatrix.shape[0], comatrix.shape[0]))
    vec = np.linalg.norm(comatrix.todense(), axis=1)
    for i in range(0, comatrix.shape[0]):
        cosinedistancemovie[i] = vec*vec[i]
        cosinedistancemovie1[i] = np.divide(dotdistancemovie[i], (cosinedistancemovie[i]))
    return (dotdistancemovie, cosinedistancemovie1)
def main():
    user()

if __name__ == "__main__": main()