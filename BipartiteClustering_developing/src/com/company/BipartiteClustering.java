package com.company;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by jessyli on 9/20/15.
 */
public class BipartiteClustering{
    public static int k1;
    public static int k2;
    public static int Biteration;
    public static HashMap<Integer, RandomAccessSparseVector> doc2w;
    public static HashMap<Integer, ArrayList<Integer>> doc2wlink;
    public static HashMap<Integer, RandomAccessSparseVector> doc2d;
    public static HashMap<Integer, ArrayList<Integer>> doc2dlink;
    public static HashMap<Integer, RandomAccessSparseVector> word2w;
    public static HashMap<Integer, ArrayList<Integer>> word2wlink;
    public static HashMap<Integer, RandomAccessSparseVector> word2d;
    public static HashMap<Integer, ArrayList<Integer>> word2dlink;

    public BipartiteClustering(HashMap<Integer, RandomAccessSparseVector> docmatrix, int k1, int k2, int Biteration, HashMap<Integer, RandomAccessSparseVector> docmatrixtrans){
        this.k1 = k1;
        this.k2 = k2;
        this.Biteration = Biteration;
        doc2w = new HashMap<Integer, RandomAccessSparseVector>();
        for(int a=0; a<k1; a++){
            RandomAccessSparseVector temp = new RandomAccessSparseVector(docmatrixtrans.get(0).size());
            doc2w.put(a,temp);
        }
        doc2wlink = new HashMap<Integer, ArrayList<Integer>>();
        for(int a=0; a<k1; a++){
            ArrayList<Integer> arrayz = new ArrayList<Integer>();
            doc2wlink.put(a, arrayz);
        }
        doc2d = new HashMap<Integer, RandomAccessSparseVector>();
        for(int a=0; a<k2; a++){
            RandomAccessSparseVector temp = new RandomAccessSparseVector(k1);
            doc2w.put(a,temp);
        }
        doc2dlink = new HashMap<Integer, ArrayList<Integer>>();
        for(int a=0; a<k2; a++){
            ArrayList<Integer> arrayz = new ArrayList<Integer>();
            doc2dlink.put(a,arrayz);
        }
        word2w = new HashMap<Integer, RandomAccessSparseVector>();
        for(int a=0; a<k1; a++){
            RandomAccessSparseVector temp = new RandomAccessSparseVector(k2);
            doc2w.put(a,temp);
        }
        word2wlink = new HashMap<Integer, ArrayList<Integer>>();
        for(int a=0; a<k1; a++){
            ArrayList<Integer> arrayz = new ArrayList<Integer>();
            word2wlink.put(a,arrayz);
        }
        word2d = new HashMap<Integer, RandomAccessSparseVector>();
        for(int a=0; a<k2; a++){
            RandomAccessSparseVector temp = new RandomAccessSparseVector(docmatrix.get(0).size());
            doc2w.put(a,temp);
        }
        word2dlink = new HashMap<Integer, ArrayList<Integer>>();
        for(int a=0; a<k2; a++){
            ArrayList<Integer> arrayz = new ArrayList<Integer>();
            word2dlink.put(a,arrayz);
        }
    }

    public static void Cluster(HashMap<Integer, RandomAccessSparseVector> docmatrix, int Biteration, HashMap<Integer, RandomAccessSparseVector> docmatrixtrans){
        int RowNumber = docmatrix.size();
        int ColumnNumber = docmatrixtrans.size();

        //word cluster and word2wcluster links
        Kmeans km = new Kmeans(k1, 0.8, docmatrix.size(), docmatrixtrans.size());
        km.run(docmatrixtrans, true, docmatrix);
        doc2w = km.centers2;
        doc2wlink = km.cluster;
        for(int b=0; b<Biteration; b++) {
            HashMap<Integer,RandomAccessSparseVector> doc2wtrans = ReadingData.TransposeMatrix(doc2w);
            Kmeans km2 = new Kmeans(k2, 0.8, doc2wtrans.size(), doc2w.size());
            km2.run(doc2wtrans,true,doc2w);
            doc2d = km2.centers1;
            doc2dlink = km2.cluster;
            //aggregate x and doc2dcluster to word2dcluster
            int count =0;
            for (int i = 0; i < k2; i++) {
                Vector tempVector;
                Vector sumVector = new RandomAccessSparseVector(docmatrixtrans.size());
                for (int j = 0; j < doc2dlink.get(i).size(); j++) {
                    tempVector = docmatrix.get(doc2dlink.get(i).get(j));
                    sumVector = tempVector.plus(sumVector);
                }
                if(doc2dlink.get(i).size()!=0) {
                    sumVector = sumVector.divide(doc2dlink.get(i).size());
                }
                else{
                    int maxcluster = 0;
                    for(int a=0; a<k2; a++){
                        if(maxcluster<doc2dlink.get(a).size()){
                            maxcluster=a;
                        }
                    }
                    if(count==(doc2dlink.get(maxcluster).size()-1)){
                        count = 0;
                    }
                    sumVector.assign(docmatrix.get(doc2dlink.get(maxcluster).get(count++)));
                }
                RandomAccessSparseVector sumvector2 = new RandomAccessSparseVector(docmatrixtrans.size());
                sumvector2.assign(sumVector);
                word2d.put(i, sumvector2);
            }

            word2dlink = doc2dlink;
            //word cluster and word2wcluster

            HashMap<Integer, RandomAccessSparseVector> word2dtrans = ReadingData.TransposeMatrix(word2d);
            Kmeans km3 = new Kmeans(k1, 0.8, word2d.size(), word2dtrans.size());
            km3.run(word2dtrans, true, word2d);
            word2w = km3.centers2;
            word2wlink = km3.cluster;
            //aggregate word2w with x to doc2w
            int count1 =0;
            for (int i = 0; i < k1; i++) {
                Vector tempVector;
                Vector sumVector = new RandomAccessSparseVector(docmatrix.size());
                for (int j = 0; j < word2wlink.get(i).size(); j++) {
                    tempVector = docmatrixtrans.get(word2wlink.get(i).get(j));
                    sumVector = tempVector.plus(sumVector);
                }
                if(word2wlink.get(i).size()!=0) {
                    sumVector = sumVector.divide(word2wlink.get(i).size());
                }
                else{
                    int maxcluster = 0;
                    for(int a=0; a<k1; a++){
                        if(maxcluster<word2wlink.get(a).size()){
                            maxcluster=a;
                        }
                    }
                    if(count1==(word2wlink.get(maxcluster).size()-1)){
                        count1 = 0;
                    }
                    sumVector.assign(docmatrixtrans.get(word2wlink.get(maxcluster).get(count1++)));
                }
                RandomAccessSparseVector sumvector2 =  new RandomAccessSparseVector(docmatrix.size());
                sumvector2.assign(sumVector);
                doc2w.put(i, sumvector2);
            }
            doc2wlink = word2wlink;
        }
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream("doc_cluster.txt"), "utf-8"));
            for(int i=0; i<word2dlink.size(); i++){
                for(int j=0; j<word2dlink.get(i).size(); j++){
                    writer.write(word2dlink.get(i).get(j)+" "+i+"\n");
                }
            }
        } catch (IOException ex) {
        } finally {
            try {writer.close();} catch (Exception ex) {/*ignore*/}
        }
        Writer writer2 = null;
        try {
            writer2 = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream("word_cluster.txt"), "utf-8"));
            for(int i=0; i<doc2wlink.size(); i++){
                for(int j=0; j<doc2wlink.get(i).size(); j++){
                    writer2.write(doc2wlink.get(i).get(j)+" "+i+"\n");
                }
            }
        } catch (IOException ex) {
        } finally {
            try {writer.close();} catch (Exception ex) {/*ignore*/}
        }

//      ################
        Writer writer3 = null;
        try {
            writer3 = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream("doc_centroid.txt"), "utf-8"));
            for(int i=0; i<word2d.size(); i++){
//                for(int j=0; j<word2d.get(i).size(); j++){
                    writer3.write(i+" "+word2d.get(i)+"\n");
//                }
            }
            System.out.println(word2d.size());
            System.out.println(word2d.get(0).size());
        } catch (IOException ex) {
        } finally {
            try {writer.close();} catch (Exception ex) {/*ignore*/}
        }
        Writer writer4 = null;
        try {
            writer4 = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream("word_centroid.txt"), "utf-8"));
            for(int i=0; i<doc2w.size(); i++){
//                 for(int j=0; j<doc2w.get(i).size(); j++){
                    writer4.write(i+" "+doc2w.get(i)+"\n");
//                }
            }
            System.out.println(doc2w.size());
            System.out.println(doc2w.get(0).size());
        } catch (IOException ex) {
        } finally {
            try {writer.close();} catch (Exception ex) {/*ignore*/}
        }


    }
}
