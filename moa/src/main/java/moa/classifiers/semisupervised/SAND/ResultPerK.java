package moa.classifiers.semisupervised;

import java.util.*;
import java.io.*;
import weka.core.Instance;
import java.util.ArrayList;

public class ResultPerK{

//    public ArrayList AllResult = new ArrayList();
//    /** Creates a new instance of Result */
//    public ResultPerK() {
//    }
//
//    public void Commit(Minstance inst)
//    {
//
//        if(inst.Comitted)
//            return;
//
//        int bin = inst.Id / 1000; //which bin to place?
//        //String debug = "Instance id = " + inst.Id +" bin = " + bin;
//
//        if(AllResult.size() <= bin)
//        {
//            int cur = AllResult.size();
//            for(int i = cur; i <= bin; i ++)
//                AllResult.add(new ResultStat());
//        }
//        ResultStat s = (ResultStat)AllResult.get(bin);
//
//        //debug += "  Allresult size = " + AllResult.size();
//
//        if(s.full())
//            System.out.println("Error - trying to insert into full bin");
//        else
//        {
//            s.addStat(inst);
//            AllResult.set(bin, s);
//            inst.Comitted = true;
//        }
//    }
}
//
//
////Save results per 1K instance
//public class ResultStat{
//    public int fp = 0;     //false positive
//    public int fn = 0;     //false negative
//    public int nc = 0;     //novel class
//    public int err = 0;    //error
//    public int total = 0;  //total instances processed
//    public boolean printed = false;
//
//    public void addStat(Minstance inst)
//    {
//        if(inst.fp)
//            fp ++;
//        else if(inst.fn)
//            fn ++;
//        if(inst.err)
//            err ++;
//        if(inst.isNovel)
//            nc ++;
//        inst.Comitted = true;
//        //String debug = inst.Id + " "+inst.fp+" "+inst.fn+" "+inst.isNovel+" "+inst.err + " "+(int)inst.EPrediction.Predclass + " "+ (int)inst.classValue()+ " "+inst.Predictions.size();
//        //Constants.logger.debug(debug);
//        //debug = "";
//        for(int i = 0; i < inst.Predictions.size(); i ++)
//        {
//            MapPrediction op = (MapPrediction)inst.Predictions.get(i);
//            //Constants.logger.debug(op.Cid+" "+op.Predclass+ " "+op.Isoutlier +" "+Math.exp(-op.Dist));
//        }
//        total ++;
//    }
//
//    public boolean full()
//    {
//        return total == 1000;
//    }
//}
