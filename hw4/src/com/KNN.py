__author__ = 'jessyli'
import os
import timeit
import sys

import numpy as np

import scipy.io
from scipy.sparse import coo_matrix
from scipy.spatial import distance
from numpy import linalg as LA
import heapq
import operator
from io import StringIO
import ReadingData

def KNNClassify(newinput,movieA, comatrix1, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie):
    #UserOrMovie == 1, Movie; UserOrMovie == 0, User
    #Distance == 1, cosine; Distance == 0, dotproduct;Distance==2, PCC
    comatrix = comatrix1.tocsr()
    if(UserOrMovie==1):
        if Distance == 1:
            temp = cosinedistancemovie
            sorted_cosine = np.zeros(k)
            for s in range(0, k):
                (index, value) = max(enumerate(temp[movieA]), key=operator.itemgetter(1))
                sorted_cosine[s]=index
                temp[movieA,index]=-3
            dominator=0.0
            v=np.zeros(k)
            cosinWeight=0.0
            cosinMean=0.0
            for a in range(0,k):
                dominator = abs(cosinedistancemovie[movieA,sorted_cosine[a]])+dominator
            for b in range(0,k):
                v[b] = cosinedistancemovie[movieA,sorted_cosine[b]]/dominator
                cosinWeight = v[b]*comatrix[sorted_cosine[b],newinput]+cosinWeight
                cosinMean = cosinMean+comatrix[sorted_cosine[b],newinput]/k
            cosinMean1 = cosinMean+3.0
            cosinWeight1 = cosinWeight+3.0
            return (cosinMean1, cosinWeight1)
        else:
            temp1 = np.squeeze(np.asarray(dotdistancemovie[movieA]))
            sorted_dot = np.zeros(k)
            for s in range(0,k):
                (index, value) = max(enumerate(temp1), key=operator.itemgetter(1))
                sorted_dot[s]=index
                temp1[index]=-1000
            dominator=0.0
            v=np.zeros(k)
            dotWeight=0.0
            dotMean=0.0
            for a in range(0,k):
                dominator = abs(dotdistancemovie[movieA,sorted_dot[a]])+dominator
            for b in range(0,k):
                v[b] = dotdistancemovie[movieA,sorted_dot[b]]/dominator
                dotWeight = v[b]*comatrix[sorted_dot[b], newinput]+dotWeight
                dotMean = dotMean+comatrix[sorted_dot[b],newinput]/k
            dotWeight1 = dotWeight+3.0
            dotMean1 = dotMean+3.0
            return (dotMean1,dotWeight1)
    else:
        if Distance == 1:
            temp = cosinedistanceuser
            sorted_cosine = np.zeros(k)
            for s in range(0, k):
                (index, value) = max(enumerate(temp[newinput]), key=operator.itemgetter(1))
                sorted_cosine[s]=index
                temp[newinput,index]=-3
            cosinMean = 0
            cosinWeight = 0
            cosinSum = 0
            for i in range(0,k):
                cosinSum = cosinSum + cosinedistanceuser[newinput,sorted_cosine[i]]
            for i in range(0, k):
                cosinMean = cosinMean+comatrix[movieA,sorted_cosine[i]]/k
                cosinWeight = cosinWeight+comatrix[movieA,sorted_cosine[i]]*cosinedistanceuser[newinput,sorted_cosine[i]]/cosinSum
            cosinMean1 = cosinMean+3.0
            cosinWeight1 = cosinWeight+3.0
            return (cosinMean1, cosinWeight1)
        elif Distance ==0:
            temp1 = dotdistanceuser[newinput,:].view(np.recarray)
            sorted_dot = np.zeros(k)
            for s in range(0,k):
                (index, value) = max(enumerate(temp1), key=operator.itemgetter(1))
                sorted_dot[s]=index
                temp1[index]=-1000
            dotMean = 0
            dotWeight = 0
            dotSum = 0
            for i in range(0,k):
                dotSum = dotSum + dotdistanceuser[newinput,sorted_dot[i]]
            for i in range(0, k):
                dotMean = dotMean + comatrix[movieA,sorted_dot[i]]/k
                dotWeight = dotWeight + comatrix[movieA,sorted_dot[i]]*dotdistanceuser[newinput,sorted_dot[i]]/dotSum
            dotMean1 = dotMean+3.0
            dotWeight1 = dotWeight+3.0
            return (dotMean1,dotWeight1)
        # elif Distance==2:
        #     # sorted_pcc = np.argsort(pccdistanceuser[newinput])[-k:]
        #     pccMean = 0
        #     pccWeight = 0
        #     pccSum = 0
        #     for i in range(0,k):
        #         # pccSum = pccSum + pccdistanceuser[newinput,sorted_pcc[i]]
        #     for i in range(0, k):
        #         pccMean = pccMean + comatrix[movieA,sorted_pcc[i]]/k
        #         pccWeight = pccWeight + comatrix[movieA,sorted_pcc[i]]*dotdistanceuser[newinput,sorted_pcc[i]]/dotpcc
        #     pccMean1 = pccMean+3.0
        #     pccWeight1 = pccWeight+3.0
        #     return (pccMean1,pccWeight1)
def main():
    start = timeit.default_timer()
    script_dir = os.path.dirname(__file__)
    rel_path = "HW4_data/train.csv"
    rel_path1 = "HW4_data/dev.csv"
    abs_file_path = os.path.join(script_dir, rel_path)
    comatrix = ReadingData.readingtrain(abs_file_path)
    abs_file_path1 = os.path.join(script_dir, rel_path1)
    comatrix1 = ReadingData.readingdev(abs_file_path1)
    (dotdistanceuser, cosinedistanceuser) = ReadingData.user()
    (dotdistancemovie, cosinedistancemovie) = ReadingData.movie()
    rel_path2 = "HW4_data/test.csv"
    abs_file_path2 = os.path.join(script_dir, rel_path2)
    comatrix2 = ReadingData.readingdev(abs_file_path2)
    (usercluster, moviecluster) = ReadingData.readcluster()
    (cosinedistancemovienew, dotdistancemovienew, cosinedistanceusernew, dotdistanceusernew) = ReadingData.readbipartite()
    # f = open('userdot10test','w')
    # f1 = open('userdot100', 'w')
    # f2 = open('userdot500', 'w')
    f3 = open('predictions.txt','w')
    # f4 = open('moviedot100', 'w')
    # f5 = open('moviedot500', 'w')
    # f6= open('usercosine10','w')
    # f7 = open('usercosine100', 'w')
    # f8 = open('usercosine500', 'w')
    # f9= open('moviecosine10','w')
    # f10 = open('moviecosine100', 'w')
    # f11= open('moviecosine500', 'w')
    # f12 = open('usercosine10weight','w')
    # f13 = open('usercosine100weight', 'w')
    # f14 = open('usercosine500weight', 'w')
    # f15 = open('moviecosine10weight','w')
    # f16 = open('moviecosine100weight', 'w')
    # f17 = open('moviecosine500weight', 'w')
    for i in range(comatrix2.shape[0]):
        movieA = comatrix2.getrow(i).todense()[0,0]
        userA = comatrix2.getrow(i).todense()[0,1]
        # movieA = moviecluster.get(movieA1)
        # userA = usercluster.get(userA1)
        # if(userA==None or movieA==None):
        #     userA=0
        #     movieA=0
        # k = 10
        # UserOrMovie = 0
        # Distance = 0
        # (dotMean, dotWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceusernew,dotdistanceusernew,cosinedistancemovienew,dotdistancemovienew)
        # f.write("%s\n" % str(dotMean))
        # k = 100
        # UserOrMovie = 0
        # Distance = 0
        # (dotMean, dotWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        # f1.write("%s\n" % str(dotWeight))
        # k = 500
        # UserOrMovie = 0
        # Distance = 0
        # (dotMean, dotWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        # f2.write("%s\n" % str(dotWeight))
        k = 10
        UserOrMovie = 1
        Distance = 0
        (dotMean, dotWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        f3.write("%s\n" % str(dotMean))
        # k = 100
        # UserOrMovie = 1
        # Distance = 0
        # (dotMean, dotWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        # f4.write("%s\n" % str(dotWeight))
        # k = 500
        # UserOrMovie = 1
        # Distance = 0
        # (dotMean, dotWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        # f5.write("%s\n" % str(dotWeight))
        # k = 10
        # UserOrMovie = 0
        # Distance = 1
        # (cosineMean, cosineWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        # f6.write("%s\n" % str(cosineMean))
        # f12.write("%s\n" % str(cosineWeight))
        # k = 100
        # UserOrMovie = 0
        # Distance = 1
        # (cosineMean,cosineWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        # f7.write("%s\n" % str(cosineMean))
        # f13.write("%s\n" % str(cosineWeight))
        # k = 500
        # UserOrMovie = 0
        # Distance = 1
        # (cosineMean,cosineWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        # f8.write("%s\n" % str(cosineMean))
        # f14.write("%s\n" % str(cosineWeight))
        # k = 10
        # UserOrMovie = 1
        # Distance = 1
        # (cosineMean,cosineWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        # f9.write("%s\n" % str(cosineMean))
        # f15.write("%s\n" % str(cosineWeight))
        # k = 100
        # UserOrMovie = 1
        # Distance = 1
        # (cosineMean,cosineWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        # f10.write("%s\n" % str(cosineMean))
        # f16.write("%s\n" % str(cosineWeight))
        # k = 500
        # UserOrMovie = 1
        # Distance = 1
        # (cosineMean,cosineWeight) = KNNClassify(userA, movieA, comatrix, k, UserOrMovie, Distance, cosinedistanceuser,dotdistanceuser,cosinedistancemovie,dotdistancemovie)
        # f11.write("%s\n" % str(cosineMean))
        # f17.write("%s\n" % str(cosineWeight))
        print(i)
    # f.close()
    # f1.close()
    # f2.close()
    f3.close()
    # f4.close()
    # f5.close()
    # f6.close()
    # f7.close()
    # f8.close()
    # f9.close()
    # f10.close()
    # f11.close()
    # f12.close()
    # f13.close()
    # f14.close()
    # f15.close()
    # f16.close()
    # f17.close()
    stop = timeit.default_timer()
    print stop - start

if __name__ == "__main__": main()