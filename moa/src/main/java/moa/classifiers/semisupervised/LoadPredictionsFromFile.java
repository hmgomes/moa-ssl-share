package moa.classifiers.semisupervised;

import com.github.javacliparser.FileOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.SemiSupervisedLearner;
import moa.core.Measurement;
import scala.Char;

import java.io.*;
import java.nio.CharBuffer;

public class LoadPredictionsFromFile extends AbstractClassifier implements SemiSupervisedLearner, MultiClassClassifier {

    public FileOption predictionsFileOption = new FileOption(
            "predictionsFile", 'f',
            "File to read predictions from.", null, "csv", true);

    BufferedReader reader;

    long numberOfPredictions = 0;
    int currentPrediction = 0;

    @Override
    public double[] getVotesForInstance(Instance inst) {
        numberOfPredictions++;
        String predictionsText = "0";
        try {
            predictionsText = reader.readLine();
        } catch(Exception e) {
            System.out.println("Cannot read another line");

            try {
                reader.close();
            } catch(Exception e2) {
                System.out.println("Could not close the predictions file.");
            }
        }

        double[] prediction = new double[inst.numClasses()];
        prediction[currentPrediction] = 1;

        if(numberOfPredictions % 100 == 0)
            System.out.println("Number of predictions :" + numberOfPredictions);

        try {
            currentPrediction = Integer.parseInt(predictionsText);
        } catch(Exception e) {
            try {
                reader.close();
            } catch(Exception e2) {
                System.out.println("Could not close the predictions file.");
            }
        }
        return prediction;
    }

    @Override
    public void resetLearningImpl() {
        File predictionsFile = this.predictionsFileOption.getFile();
        if (predictionsFile != null) {
            try {
                if (predictionsFile.exists()) {
                    reader = new BufferedReader(new FileReader(predictionsFile));
                    String predictionsText = reader.readLine();
                    currentPrediction = Integer.parseInt(predictionsText);
                } else {
                    System.out.println("Predictions file does not exists");
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open predictions file " + predictionsFile, ex);
            }
        }



    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public int trainOnUnlabeledInstance(Instance instance) {
        return 0;
    }

    @Override
    public void addInitialWarmupTrainingInstances() {
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }
}
