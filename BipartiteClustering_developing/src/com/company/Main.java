
package com.company;
import org.apache.mahout.math.*;
import java.util.HashMap;
/**
 * Created by jessyli on 9/20/15.
 * Run the main.java and the document of doc clustering file and
 * word clustering file would be created in the catalog of this project.
 * sum of doc similarity and word similarity would be printed out.
 */
public class Main {

    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        int k1=539;
        int k2=1092;
        int Biteration = 20;
        HashMap<Integer, RandomAccessSparseVector> docmatrix;
        //these two similar lines are written for develop data set and testing data set. The first line is
        //for develop data set, the second line is for testing data set.
        docmatrix = ReadingData.readFileByLines("train.csv");

//        docmatrix = ReadingData.readFileByLines("HW2_test.docVectors", 942, 13924, "HW2_test.df");
        HashMap<Integer, RandomAccessSparseVector> docmatrixtrans = ReadingData.TransposeMatrix(docmatrix);
        BipartiteClustering bc = new BipartiteClustering(docmatrix, k1, k2, Biteration, docmatrixtrans);
        bc.Cluster(docmatrix, Biteration, docmatrixtrans);
        //doc cosine similarity
        double sumdistance = 0;
        for(int j=0; j<k2; j++){
            for(int i=0; i<bc.word2dlink.get(j).size(); i++){

                double tempdistance = docmatrix.get(bc.word2dlink.get(j).get(i)).dot(bc.word2d.get(j));
                tempdistance = tempdistance/(docmatrix.get(bc.word2dlink.get(j).get(i)).norm(2)*bc.word2d.get(j).norm(2));
                sumdistance = tempdistance+sumdistance;
            }
        }
//        System.out.println("doc similarity"+sumdistance);
        //word cosine similarity
        double sumdistance2 = 0;
        for(int j=0; j<k1; j++){
            for(int i=0; i<bc.doc2wlink.get(j).size(); i++){

                double tempdistance2 = docmatrixtrans.get(bc.doc2wlink.get(j).get(i)).dot(bc.doc2w.get(j));
                tempdistance2 = tempdistance2/(docmatrixtrans.get(bc.doc2wlink.get(j).get(i)).norm(2)*bc.doc2w.get(j).norm(2));
                sumdistance2 = tempdistance2+sumdistance2;
            }
        }
//        System.out.println("word similarity"+sumdistance2);
        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println(totalTime);
    }


}
