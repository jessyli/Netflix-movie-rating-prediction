package com.company;
import org.apache.mahout.math.RandomAccessSparseVector;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
/**
 * Created by jessyli on 9/13/15.
 */
public class ReadingData {

    public static HashMap readFileByLines(String fileName1) {
        HashMap<Integer, RandomAccessSparseVector> docmatrix = new HashMap<Integer, RandomAccessSparseVector>();
        int sum = 0;
        int count = 0;
        File file = new File(fileName1);
        BufferedReader reader = null;

        try {
            reader = new BufferedReader(new FileReader(file));
            String tempString = null;
            int line = 0;
            int key = 0;
            RandomAccessSparseVector tempvector = new RandomAccessSparseVector(10916);
            while ((tempString = reader.readLine()) != null) {
                String[] blocks = tempString.split(",");
                int block_key = Integer.parseInt(blocks[0]);
                if(block_key==key){
                    int blockcol = Integer.parseInt(blocks[1]);
                    int value = Integer.parseInt(blocks[2]);

                    if(block_key==3) {
                        count = count + 1;
                        sum = sum+value;
                    }
                    tempvector.set(blockcol,value);
                }
                else{
                    docmatrix.put(key, tempvector);
                    if(block_key!=key+1){
                        tempvector = new RandomAccessSparseVector(10916);
                        key=key+1;
                        docmatrix.put(key, tempvector);
                    }
                    else{
                        key = block_key;
                        tempvector = new RandomAccessSparseVector(10916);

                        int blockcol = Integer.parseInt(blocks[1]);
                        int value = Integer.parseInt(blocks[2]);
                        tempvector.set(blockcol, value);
                    }

                }

            }
//            System.out.println(sum/count);
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return docmatrix;
    }
    public static HashMap<Integer, RandomAccessSparseVector> TransposeMatrix(HashMap<Integer, RandomAccessSparseVector> docmatrix){
        HashMap<Integer, RandomAccessSparseVector> docmatrixTrans = new HashMap<Integer, RandomAccessSparseVector>();
        for(int i=0; i<docmatrix.size(); i++){
            for(int j=0; j<docmatrix.get(0).size(); j++){
                if(!docmatrixTrans.containsKey(j)){
                    RandomAccessSparseVector temv = new RandomAccessSparseVector(docmatrix.size());
                    docmatrixTrans.put(j, temv);
                }
                docmatrixTrans.get(j).set(i, docmatrix.get(i).get(j));

//                docmatrixTrans.get(j).set(i, 1);

            }
        }
        return  docmatrixTrans;
    }


}
