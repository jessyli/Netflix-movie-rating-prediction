package com.company;

import org.apache.mahout.math.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by jessyli on 9/13/15.
 */
public class Kmeans {
    static int k; // number of clusters
    static HashMap<Integer, RandomAccessSparseVector> centers1; // store the centers in last iteration(row)
    static HashMap<Integer, RandomAccessSparseVector> centers2; // store the centers in last iteration(column)
    static HashMap<Integer, ArrayList<Integer>> cluster; //store index of the classified cluster for each vector
    static double err;// sign of stop

    public Kmeans(int k, double err, int rownumber, int columnumber){
        this.k = k;
        this.err = err;
        centers1 = new HashMap<Integer, RandomAccessSparseVector>();
        for(int i=0; i<k; i++){
            RandomAccessSparseVector temp1 = new RandomAccessSparseVector(columnumber);
            centers1.put(i, temp1);
        }
        centers2 = new HashMap<Integer, RandomAccessSparseVector>();
        for(int j=0; j<k; j++){
            RandomAccessSparseVector temp2 = new RandomAccessSparseVector(rownumber);
            centers2.put(j, temp2);
        }
    }

    public static void run(HashMap<Integer, RandomAccessSparseVector> docmatrix, boolean DocOrTerm, HashMap<Integer, RandomAccessSparseVector> docmatrixtrans){
        init(docmatrix, DocOrTerm,docmatrixtrans);
        classify(docmatrix, DocOrTerm, docmatrixtrans);
        boolean stopsign = NewCenter(docmatrix, DocOrTerm, docmatrixtrans);
        int count=0;
        while(!stopsign && count<8){
            count++;
            classify(docmatrix, DocOrTerm, docmatrixtrans);
            stopsign = NewCenter(docmatrix, DocOrTerm, docmatrixtrans);
        }
    }
    //initalize the clusters
    public static void init(HashMap<Integer, RandomAccessSparseVector> docmatrix, boolean DocOrTerm, HashMap<Integer, RandomAccessSparseVector> docmatrixtrans){
        int RowNumber = docmatrix.size();
        int ColumnNumber = docmatrix.get(0).size();
        Random rand = new Random();
        int[] kcenter = new int[k];
        // if DocOrTerm == true, do kmeans by doc, otherwise, do kmeans by term
        if(DocOrTerm == true) {
            ArrayList<Integer> checkrandom = new ArrayList<Integer>();
            for(int i=0; i< k; i++){
                kcenter[i] = rand.nextInt(RowNumber-1);
                while(checkrandom.contains(kcenter[i])){
                    kcenter[i] = rand.nextInt(RowNumber-1);
                }
                checkrandom.add(kcenter[i]);
                centers1.put(i, docmatrix.get(kcenter[i]));
            }
        }
        else{
            ArrayList<Integer> checkrandom = new ArrayList<Integer>();
            for(int i=0; i< k; i++){
                kcenter[i] = rand.nextInt(ColumnNumber-1);
                while(checkrandom.contains(kcenter[i])){
                    kcenter[i] = rand.nextInt(ColumnNumber-1);
                }
                    checkrandom.add(kcenter[i]);
                    centers2.put(i, docmatrixtrans.get(kcenter[i]));
            }
        }

    }

    // classify each vector into k clusters
    public static void classify(HashMap<Integer, RandomAccessSparseVector> docmatrix, boolean DocOrTerm, HashMap<Integer, RandomAccessSparseVector> docmatrixtrans){
        cluster = new HashMap<Integer, ArrayList<Integer>>();
        for(int a=0; a<k; a++){
            ArrayList<Integer> arrayz = new ArrayList<Integer>();
            cluster.put(a,arrayz);
        }
        if(DocOrTerm == true) {
            int RowNumber = docmatrix.size();
            for (int i = 0; i < RowNumber; i++) {
                int index = 0;
                double MaxDistance = 0;
                for (int j = 0; j < k; j++) {
                    double tempDistance = docmatrix.get(i).dot(centers1.get(j));
                    tempDistance = tempDistance/(docmatrix.get(i).norm(2)*centers1.get(j).norm(2));
                    if (tempDistance > MaxDistance) {
                        MaxDistance = tempDistance;
                        index = j;
                    }
                }
                for(int z=0; z<k; z++){
                    if(index == z){
                            cluster.get(z).add(i);
                    }
                }
            }
        }
        else{
            int ColumnNumber = docmatrixtrans.size();
            for (int i = 0; i < ColumnNumber; i++) {
                int index = 0;
                double MaxDistance = 0;
                for (int j = 0; j < k; j++) {
                    double tempDistance = docmatrixtrans.get(i).dot(centers2.get(j));
                    tempDistance = tempDistance/(docmatrixtrans.get(i).norm(2)*centers2.get(j).norm(2));
                    if (MaxDistance < tempDistance) {
                        MaxDistance = tempDistance;
                        index = j;
                    }
                }
                for(int z=0; z<k; z++){
                    if(index == z){
                            cluster.get(z).add(i);
                    }
                }
            }
        }
    }
    // recalculate the center for each cluster and determine whether this iteration could stop
    public static boolean NewCenter(HashMap<Integer, RandomAccessSparseVector> docmatrix, boolean DocOrTerm, HashMap<Integer, RandomAccessSparseVector> docmatrixtrans){
        if(DocOrTerm==true){
            int RowNumber = docmatrix.size();
            int ColumnNumber = docmatrix.get(0).size();
            boolean StopSign=true;
            int count=0;
            int stopcount = 0;
            for(int i=0; i<k; i++){
                Vector tempVector;
                Vector sumVector = new RandomAccessSparseVector(ColumnNumber);
                for(int j=0; j<cluster.get(i).size(); j++){
                    tempVector = docmatrix.get(cluster.get(i).get(j));
                    sumVector = tempVector.plus(sumVector);
                }
                if(cluster.get(i).size()!=0){
                    sumVector = sumVector.divide(cluster.get(i).size());
                }
                else{
                    int maxcluster = 0;
                    for(int a=0; a<k; a++){
                        if(maxcluster<cluster.get(a).size()){
                            maxcluster=a;
                        }
                    }
                    if(count==cluster.get(maxcluster).size()){
                        count = 0;
                    }
                    sumVector.assign(docmatrix.get(cluster.get(maxcluster).get(count++)));
                }
                centers1.get(i).assign(sumVector);
                double tempDistance = sumVector.dot(centers1.get(i));
                tempDistance = tempDistance/(sumVector.norm(2)*centers1.get(i).norm(2));
                if(tempDistance>=err){
                    stopcount++;
                }
            }
            if((stopcount/k)>0.8 && stopcount!=0){
                StopSign=true;
            }
            else{
                StopSign=false;
            }
            return StopSign;
        }
        else{
            int count =0;
            int RowNumber = docmatrixtrans.get(0).size();
            int ColumnNumber = docmatrixtrans.size();
            int stopcount = 0;
            boolean StopSign=true;
            for(int i=0; i<k; i++){
                Vector tempVector ;
                Vector sumVector = new RandomAccessSparseVector(RowNumber);
                for(int j=0; j<cluster.get(i).size(); j++){
                    tempVector = docmatrixtrans.get(cluster.get(i).get(j));
                    sumVector = tempVector.plus(sumVector);
                }
                if(cluster.get(i).size()!=0){
                    sumVector = sumVector.divide(cluster.get(i).size());
                }
                else{
                    int maxcluster = 0;
                    for(int a=0; a<k; a++){
                        if(maxcluster<cluster.get(a).size()){
                            maxcluster=a;
                        }
                    }
                    if(count==(cluster.get(maxcluster).size()-1)){
                        count = 0;
                    }
                    sumVector.assign(docmatrixtrans.get(cluster.get(maxcluster).get(count++)));
                }
                centers2.get(i).assign(sumVector);
                double tempDistance = sumVector.dot(centers2.get(i));
                tempDistance = tempDistance/(sumVector.norm(2)*centers2.get(i).norm(2));
                if(tempDistance>=err){
                    stopcount++;
                }
            }
            if(stopcount/k>0.8 && stopcount!=0){
                StopSign=true;
            }
            else{
                StopSign=false;
            }
            return StopSign;
        }
    }
}
