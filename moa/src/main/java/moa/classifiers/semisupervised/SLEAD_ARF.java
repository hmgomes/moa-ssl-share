package moa.classifiers.semisupervised;

import com.github.javacliparser.*;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.SemiSupervisedLearner;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.meta.AdaptiveRandomForest;
import moa.core.*;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.options.ClassOption;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;

public class SLEAD_ARF extends AbstractClassifier implements SemiSupervisedLearner, MultiClassClassifier {

    public ClassOption baseEnsembleOption = new ClassOption("baseEnsemble", 'l',
            "Ensemble to train on instances.", AdaptiveRandomForest.class, "AdaptiveRandomForest");

    public MultiChoiceOption SSLStrategyOption = new MultiChoiceOption("SSLStrategy", 'p',
            "Defines the strategy to cope with unlabeled instances, i.e. the semi-supervised learning (SSL) strategy.",
            new String[]{"MultiViewMinKappa", "MultiViewMinKappaConfidence"},
            new String[]{"MultiViewMinKappa", "MultiViewMinKappaConfidence"}, 0);

    public FloatOption SSL_minConfidenceOption = new FloatOption("SSL_minConfidence", 'm',
            "Minimum confidence to use unlabeled instance to train.", 0.60, 0.0, 1.0);

    public MultiChoiceOption weightFunctionOption = new MultiChoiceOption("weightFunction", 'w',
            "Defines the function to weight pseudo-labeled instances. ",
            new String[]{"Constant1", "Confidence", "ConfidenceWeightShrinkage", "UnsupervisedDetectionWeightShrinkage"},
            new String[]{"Constant1", "Confidence", "ConfidenceWeightShrinkage", "UnsupervisedDetectionWeightShrinkage"}, 2);

    public MultiChoiceOption pairingFunctionOption = new MultiChoiceOption("pairingFunction", 't',
            "Defines the function to pair learners, default is minimum Kappa",
            new String[]{"MinKappa", "Random", "MajorityTrainsMinority"},
            new String[]{"MinKappa", "Random", "MajorityTrainsMinority"}, 0);

    public FloatOption SSL_weightShrinkageOption = new FloatOption("SSL_weightShrinkageOption", 'n',
            "Minimum confidence to use unlabeled instance to train.", 100.0, 0.0, Integer.MAX_VALUE);

    public FlagOption pseudoLabelChangeDetectionUpdateAllowedOption = new FlagOption("pseudoLabelChangeDetectionUpdateAllowed", 'c',
            "Allow pseudo-labeled instances to be used to update the change detector");


    public FileOption debugOutputFileOption = new FileOption("debugOutputFile", 'o',
            "File to append debug information.", null, "csv", true);

    public FlagOption debugPerfectPseudoLabelingOption = new FlagOption("debugPerfectPseudoLabeling", 'd',
            "CAUTION: THIS OVERRIDES THE PSEUDO-LABELING STRATEGY AND ALWAYS CHOOSE THE CORRECT LABEL AS THE PSEUDO-LABEL, FOR DEBUG ONLY!");


    public ClassOption studentLearnerForUnsupervisedDriftDetectionOption = new ClassOption("studentLearnerForUnsupervisedDriftDetection", 'g',
            "Student to mimic the ensemble. It is used for unsupervised drift detection.", Classifier.class, "trees.HoeffdingTree -g 50 -c 0.01");

    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-5");

    public IntOption unsupervisedDetectionWeightWindowOption = new IntOption("unsupervisedDetectionWeightWindow", 'z',
            "The labeledInstancesBuffer lenght of the sigmoid functions for the unsupervised drift detection pseudo-labeling weighting.", 20, 0, Integer.MAX_VALUE);


    public FlagOption debugShowTTTAccuracyConfidenceOption = new FlagOption("debugShowTTTAccuracyConfidence", 'v',
            "Writes to the output after every <debugFrequency> instances the accuracy " +
                    "and avg confidence in incorrect predictions and avg confidence in correct predictions");

    AdaptiveRandomForest baseEnsemble;

    Classifier student;

    protected ChangeDetector driftDetectionMethod;


    protected long lastDriftDetectedAt;
    // For debug only.
    protected long lastDriftDetectedAtUnlabelled;
    public WindowClassificationPerformanceEvaluator evaluatorSupervised;

    protected C[][] pairsOutputsKappa;
//    protected ArrayList<Integer> seedsKappa;
//    protected int[] subnetworksKappa;

    // Statistics
    protected long instancesSeen; // count only labeled (updated only on trainOnInstance()
    protected long allInstancesSeen; // Unlabeled and labeled
    protected long instancesPseudoLabeled;
    protected long instancesCorrectPseudoLabeled;
    protected double currentAverageKappa;
    protected double currentMinKappa;
    protected double currentMaxKappa;
    protected int numberOfUnsupervisedDriftsDetected;

    public static final int SSL_MULTI_VIEW_MIN_KAPPA = 0;
    public static final int SSL_MULTI_VIEW_MIN_KAPPA_CONFIDENCE = 1;

    public static final int SSL_WEIGHT_CONSTANT_1 = 0;
    public static final int SSL_WEIGHT_CONFIDENCE = 1;
    public static final int SSL_WEIGHT_CONFIDENCE_WS = 2;
    public static final int SSL_WEIGHT_UNSUPERVISED_DETECTION_WEIGHT_SHRINKAGE = 3;   // UnsupervisedDetectionWeightShrinkage

    public static final int SSL_PAIRING_MIN_KAPPA = 0;
    public static final int SSL_PAIRING_RANDOM = 1;

    PrintStream outputDebugStream = null;


    private double sigmoid(long t, long t0, int window) {
        return 1/(1 + Math.exp(-4*(t-t0)/window));
    }

    private double weightAge(long createdOn, long lstDriftDetectedOn, long labelledInstancesSeen, int window) {
        // lastDrift
        // instancesSeen (labelled data seen so far)
        double prob_drift = sigmoid(createdOn, lstDriftDetectedOn, window);
        double prob_current = 1 - sigmoid(createdOn, labelledInstancesSeen - window, window);

        return prob_drift <= prob_current ? prob_drift : prob_current;
    }

    private void estimateSupervisedLearningAccuracy(Instance instance) {
        if(this.baseEnsemble != null) {
            InstanceExample example = new InstanceExample(instance);
            double[] votes = this.baseEnsemble.getVotesForInstance(instance);
            this.evaluatorSupervised.addResult(example, votes);
        }
    }

    private boolean trainAndUpdateStudentDetector(Instance instance) {
        double[] ensembleVotes = this.baseEnsemble.getVotesForInstance(instance);
        int ensemblePrediction = Utils.maxIndex(ensembleVotes);

//        if(this.allInstancesSeen % 100 == 0)
//            System.out.println("Ok, " + this.allInstancesSeen);

        Instance instanceWithEnsemblePrediction = instance.copy();
        instanceWithEnsemblePrediction.setClassValue(ensemblePrediction);

        this.student.trainOnInstance(instanceWithEnsemblePrediction);
        boolean correctlyClassifies = this.student.correctlyClassifies(instanceWithEnsemblePrediction);

        /*********** drift detection ***********/
        // Update the DRIFT detection method
        this.driftDetectionMethod.input(correctlyClassifies ? 0 : 1);
        // Check if there was a change
        if (this.driftDetectionMethod.getChange()) {
            this.numberOfUnsupervisedDriftsDetected++;
            // There was a change, this model must be reset
            // this.reset(instance, instancesSeen, random);
//            System.out.println(this.allInstancesSeen + ", " + this.instancesSeen + ", "+ this.evaluatorSupervised.getPerformanceMeasurements()[1].getValue() +", [SLEAD] DRIFT DETECTED");

            // should change order?
            this.student = (Classifier) getPreparedClassOption(this.studentLearnerForUnsupervisedDriftDetectionOption);
            this.student.resetLearning();

            this.driftDetectionMethod = (ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption);
            this.driftDetectionMethod.resetLearning();

//            if(this.instancesSeen % 450 == 0){
//            for(int i = 0 ; i < this.baseEnsemble.getEnsembleMembers().length ; ++i)
//                System.out.print("~~~" + (this.instancesSeen - this.baseEnsemble.getEnsembleMembers()[i].createdOn) + " (" + getAgeSinceLastDrift(i) + "),");
//            System.out.println();
//            }

//            for(int i = 0 ; i < this.baseEnsemble.getEnsembleMembers().length ; ++i) {
//                this.baseEnsemble.getEnsembleMembers()[i].evaluator.getPerformanceMeasurements()[1].getValue();
//            }


            this.lastDriftDetectedAt = this.instancesSeen;
            this.lastDriftDetectedAtUnlabelled = this.allInstancesSeen;

            return true;
        }
        return false;
    }


    @Override
    public void resetLearningImpl() {
        this.baseEnsemble = (AdaptiveRandomForest) getPreparedClassOption(this.baseEnsembleOption);
        this.baseEnsemble.resetLearning();

        this.student = (Classifier) getPreparedClassOption(this.studentLearnerForUnsupervisedDriftDetectionOption);
        this.student.resetLearning();
        this.lastDriftDetectedAt = 0;

        this.driftDetectionMethod = (ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption);
        this.driftDetectionMethod.resetLearning();

        this.instancesSeen = 0;
        this.allInstancesSeen = 0;
        this.instancesCorrectPseudoLabeled = 0;
        this.instancesPseudoLabeled = 0;
        this.currentAverageKappa = 0.0;
        this.currentMaxKappa = -1.0;
        this.currentMinKappa = 1.0;

        this.numberOfUnsupervisedDriftsDetected = 0;

        this.evaluatorSupervised = new WindowClassificationPerformanceEvaluator();
        this.evaluatorSupervised.widthOption.setValue(1000);
        this.evaluatorSupervised.reset();

        //File for debug
        File outputDebugFile = this.debugOutputFileOption.getFile();
        if (outputDebugFile != null) {
            try {
                if (outputDebugFile.exists()) {
                    outputDebugStream = new PrintStream(
                            new FileOutputStream(outputDebugFile, true), true);
                } else {
                    outputDebugStream = new PrintStream(
                            new FileOutputStream(outputDebugFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open debug file: " + outputDebugFile, ex);
            }
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        ++this.instancesSeen;
        ++this.allInstancesSeen;

//        estimateSupervisedLearningAccuracy(instance);

        if (this.baseEnsemble.getEnsembleMembers() == null) {
            this.baseEnsemble.initEnsemble(instance);
        }

        // TODO: add methods to access base models for other ensembles
        AdaptiveRandomForest.ARFBaseLearner[] ensembleMembers = this.baseEnsemble.getEnsembleMembers();
//        Classifier[] ensembleMembers = this.baseEnsemble.getSubClassifiers();

        int [][] changeStatus = kappaUpdateFirst(instance, ensembleMembers);

        // Store the predictions of each learner.
        HashMap<Integer, Integer> classifiersExactPredictions = classifiersPredictions(instance, ensembleMembers);

        /********* TRAIN THE ENSEMBLE *********/
        this.baseEnsemble.trainOnInstanceImpl(instance);
        /********* TRAIN THE ENSEMBLE *********/

        kappaUpdateSecond(changeStatus, ensembleMembers, classifiersExactPredictions);


        trainAndUpdateStudentDetector(instance);

    }

    @Override
    public void addInitialWarmupTrainingInstances() {
        // TODO: add counter, but this may not be necessary for this class
    }

    @Override
    public int trainOnUnlabeledInstance(Instance instance) {
        this.allInstancesSeen++;

//        estimateSupervisedLearningAccuracy(instance);

        if(debugPerfectPseudoLabelingOption.isSet() && instance.classIsMissing()) {
            throw new RuntimeException("Pseudo-labeling debug is on, but the true class is not available");
        }

        int predictedIndex = -1;

        // TODO: add methods to access base models for other ensembles
        AdaptiveRandomForest.ARFBaseLearner[] ensembleMembers = this.baseEnsemble.getEnsembleMembers();

        int [][] changeStatus = kappaUpdateFirst(instance, ensembleMembers);

////         Store the predictions of each learner.
        HashMap<Integer, Integer> classifiersExactPredictions = classifiersPredictions(instance, ensembleMembers);

        boolean changeDetectionUpdateAllowed = this.pseudoLabelChangeDetectionUpdateAllowedOption.isSet();

        if(this.pairsOutputsKappa == null)
            throw new RuntimeException("No kappa network for MULTI_VIEW_MIN_KAPPA_CONFIDENCE ");

        ArrayList<Integer> indicesNoReposition = new ArrayList<>();
        for(int i = 0 ; i < ensembleMembers.length ; ++i)
            indicesNoReposition.add(i);

        for(int i = 0 ; i < ensembleMembers.length ; ++i) {

            int idxMinKappa = -1;
            switch(this.pairingFunctionOption.getChosenIndex()) {
                case SSL_PAIRING_MIN_KAPPA:
                    for (int j = 0; j < ensembleMembers.length; ++j) {
                        if (j != i) {
                            // Find the smallest kappa, i.e., far from
                            double currentKappa = this.getKappa(i, j);
                            if (idxMinKappa == -1 || currentKappa < this.getKappa(i, idxMinKappa))
                                idxMinKappa = j;
                        }
                    }
                    break;
                case SSL_PAIRING_RANDOM:
                    // keep randomly choosing until idxMinKappa != i
                    do {
                        idxMinKappa = Math.abs(this.classifierRandom.nextInt()) % ensembleMembers.length;
                    } while(idxMinKappa == i);
                    break;
            }


            double[] votesRawMinKappa = ensembleMembers[idxMinKappa].getVotesForInstance(instance);
            DoubleVector votesMinKappa = new DoubleVector(votesRawMinKappa);
            int predictedIndexByMinKappa = votesMinKappa.maxIndex();

            Instance instanceWithPredictedLabelByMinKappa = instance.copy();

            /*** DEBUG PURPOSES ONLY, ON REAL TESTS, THIS SHOULD NEVER BE USED ***/
            if (this.debugPerfectPseudoLabelingOption.isSet())
                instanceWithPredictedLabelByMinKappa.setClassValue(instance.classValue());
            else
                instanceWithPredictedLabelByMinKappa.setClassValue(predictedIndexByMinKappa);

            double minKappaIdxEstimatedAccuracy = ensembleMembers[idxMinKappa].evaluator.getPerformanceMeasurements()[1].getValue() / 100.0;

            // Sanity check...
            if (predictedIndexByMinKappa >= 0 && votesMinKappa.sumOfValues() > 0.0 && minKappaIdxEstimatedAccuracy > 0.0) {

                double weightMinKappa = -1;

                switch(weightFunctionOption.getChosenIndex()) {
                    case SSL_WEIGHT_CONSTANT_1:
                        weightMinKappa = 1;
                        break;
                    case SSL_WEIGHT_CONFIDENCE:
                        weightMinKappa = votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues();
                        break;
                    case SSL_WEIGHT_CONFIDENCE_WS:
                        weightMinKappa = votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues() / this.SSL_weightShrinkageOption.getValue();
                        break;

                    case SSL_WEIGHT_UNSUPERVISED_DETECTION_WEIGHT_SHRINKAGE:
                        // weight = confidence on prediction * weight_age()
//                        weightMinKappa = votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues()
//                                / ensembleMembers[i].createdOn;

                        weightMinKappa = votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues()
                                * weightAge(ensembleMembers[idxMinKappa].createdOn, this.lastDriftDetectedAt, this.instancesSeen,
                                this.unsupervisedDetectionWeightWindowOption.getValue());
                        break;
                }

//                        votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues() / this.SSL_weightShrinkageOption.getValue();
//                        instance.weight() / this.SSL_weightShrinkageOption.getValue();

                switch (SSLStrategyOption.getChosenIndex()) {
                    case SSL_MULTI_VIEW_MIN_KAPPA:
                        /***** TRAIN MEMBER i WITH PSEUDO-LABELED INSTANCE *****/
                        ensembleMembers[i].trainOnInstance(
                                instanceWithPredictedLabelByMinKappa, weightMinKappa, this.instancesSeen, this.classifierRandom, changeDetectionUpdateAllowed);

                        if (!instance.classIsMissing() && ((int) instanceWithPredictedLabelByMinKappa.classValue()) == ((int) instance.classValue()))
                            ++this.instancesCorrectPseudoLabeled;
                        ++this.instancesPseudoLabeled;

                        break;
                    case SSL_MULTI_VIEW_MIN_KAPPA_CONFIDENCE:
                        double confidenceByMinKappa = votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues();
                        DoubleVector votesByCurrent = new DoubleVector(ensembleMembers[i].getVotesForInstance(instance));
                        int predictedByCurrent = votesByCurrent.maxIndex();
                        double confidenceByCurrent = votesByCurrent.getValue(predictedByCurrent) / votesByCurrent.sumOfValues();

                        if (confidenceByMinKappa > this.SSL_minConfidenceOption.getValue() &&
                                predictedByCurrent != predictedIndexByMinKappa &&
                                confidenceByMinKappa > confidenceByCurrent) {


                            /***** TRAIN MEMBER i WITH PSEUDO-LABELED INSTANCE *****/
                            ensembleMembers[i].trainOnInstance(
                                    instanceWithPredictedLabelByMinKappa, weightMinKappa, this.instancesSeen, this.classifierRandom, changeDetectionUpdateAllowed);

                            if (!instance.classIsMissing() && ((int) instanceWithPredictedLabelByMinKappa.classValue()) == ((int) instance.classValue()))
                                ++this.instancesCorrectPseudoLabeled;
                            ++this.instancesPseudoLabeled;
                        }
                        break;
                }
            }
        }

        // Check if changes occurred.
        kappaUpdateSecond(changeStatus, ensembleMembers, classifiersExactPredictions);


        trainAndUpdateStudentDetector(instance);

        return predictedIndex;
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        if (this.baseEnsemble.getEnsembleMembers() == null) {
            this.baseEnsemble.initEnsemble(instance);
        }

        estimateSupervisedLearningAccuracy(instance);

        return this.baseEnsemble.getVotesForInstance(instance);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        int totalDriftResets = 0;
        for(int i = 0 ; i < this.baseEnsemble.getEnsembleMembers().length ; ++i)
            totalDriftResets += this.baseEnsemble.getEnsembleMembers()[i].numberOfDriftsDetected;


        if(this.debugShowTTTAccuracyConfidenceOption.isSet())
//            if(this.allInstancesSeen % this.debugFrequencyOption.getValue() == 0)
            System.out.println(this.allInstancesSeen + "," + this.instancesSeen + "," +
                    this.evaluatorSupervised.getPerformanceMeasurements()[1].getValue()); //+ ", avg conf incorr ???, avg conf corr ???");

        return new Measurement[]{
                new Measurement("#pseudo-labeled", this.instancesPseudoLabeled),
                new Measurement("#correct pseudo-labeled", this.instancesCorrectPseudoLabeled),
                new Measurement("accuracy pseudo-labeled", this.instancesCorrectPseudoLabeled / (double) this.instancesPseudoLabeled * 100),
                new Measurement("#drift resets", totalDriftResets),
                new Measurement("avg kappa", this.currentAverageKappa),
                new Measurement("max kappa", this.currentMaxKappa),
                new Measurement("min kappa", this.currentMinKappa),
                new Measurement("#unsup drifts", this.numberOfUnsupervisedDriftsDetected),
                new Measurement("latest detection", this.lastDriftDetectedAtUnlabelled)
//                new Measurement("accuracy supervised learner", this.evaluatorSupervised.getPerformanceMeasurements()[1].getValue())
        };
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return true;
    }


    private int[][] kappaUpdateFirst(Instance instance, AdaptiveRandomForest.ARFBaseLearner[] ensembleMembers) {
//    private int[][] kappaUpdateFirst(Instance instance, Classifier[] ensembleMembers) {
        /******************* K STATISTIC *******************/
        if (this.pairsOutputsKappa == null) {
            this.pairsOutputsKappa = new C[ensembleMembers.length][ensembleMembers.length];
            for (int i = 0; i < this.pairsOutputsKappa.length; ++i) {
                for (int j = 0; j < this.pairsOutputsKappa[i].length; ++j) {
                    this.pairsOutputsKappa[i][j] = new C(instance.numClasses());
                }
            }
        }

        // Save the number of drifts detected per base model
        int driftStatus[] = new int[ensembleMembers.length];
        int warningStatus[] = new int[ensembleMembers.length];

        for (int i = 0; i < driftStatus.length; ++i) {
            driftStatus[i] = ensembleMembers[i].numberOfDriftsDetected;
            warningStatus[i] = ensembleMembers[i].numberOfWarningsDetected;
        }

        return new int[][]{driftStatus, driftStatus};
    }

    private void kappaUpdateSecond(int[][] changeStatus, AdaptiveRandomForest.ARFBaseLearner[] ensembleMembers,
                                   HashMap<Integer, Integer> classifiersExactPredictions) {
        /******************* K STATISTIC *******************/
        int currentDriftStatus[] = new int[ensembleMembers.length];
        int currentWarningStatus[] = new int[ensembleMembers.length];
        // Compare the number of drifts detected per base model
        for (int i = 0; i < changeStatus[0].length; ++i) {
            currentDriftStatus[i] = ensembleMembers[i].numberOfDriftsDetected;
            currentWarningStatus[i] = ensembleMembers[i].numberOfWarningsDetected;
        }

        for (int i = 0; i < ensembleMembers.length; ++i) {
            for (int j = i + 1; j < ensembleMembers.length; ++j) {
                if (currentDriftStatus[i] - changeStatus[0][i] == 0)
                    this.pairsOutputsKappa[i][j].update(classifiersExactPredictions.get(i),
                            classifiersExactPredictions.get(j));
                else
                    this.pairsOutputsKappa[i][j].reset();
            }
        }

        double sumKappa = 0.0;
        this.currentMinKappa = 1.0;
        this.currentMaxKappa = -1.0;
        double countKappa = (double) (ensembleMembers.length * (ensembleMembers.length - 1) / 2);
        for (int i = 0; i < ensembleMembers.length; ++i) {
            for (int j = i + 1; j < ensembleMembers.length; ++j) {
                double kappa = this.pairsOutputsKappa[i][j].k();
                if(kappa < this.currentMinKappa)
                    this.currentMinKappa = kappa;
                if(kappa > this.currentMaxKappa)
                    this.currentMaxKappa = kappa;
                sumKappa += Double.isNaN(kappa) ? 1.0 : kappa;
            }
        }
        this.currentAverageKappa = sumKappa / countKappa;
    }

    private HashMap<Integer, Integer> classifiersPredictions(Instance instance, AdaptiveRandomForest.ARFBaseLearner[] ensembleMembers) {
        // Store the predictions of each learner.
        HashMap<Integer, Integer> classifiersExactPredictions = new HashMap<>();
        for (int i = 0; i < ensembleMembers.length; i++) {
            double[] rawVote = ensembleMembers[i].getVotesForInstance(instance);
            DoubleVector vote = new DoubleVector(rawVote);

            int maxIndex = vote.maxIndex();
            if (maxIndex < 0) {
                maxIndex = this.classifierRandom.nextInt(instance.numClasses());
            }
            classifiersExactPredictions.put(i, maxIndex);
        }
        return classifiersExactPredictions;
    }

    protected double getKappa(int i, int j) {
        assert(i != j);
        if(this.pairsOutputsKappa == null)
            return -6;
        return i > j ? this.pairsOutputsKappa[j][i].k() : this.pairsOutputsKappa[i][j].k();
    }

    // For Kappa calculation
    protected class C{
        private final long[][] C;
        private final int numClasses;
        private int instancesSeen = 0;

        public C(int numClasses) {
            this.numClasses = numClasses;
            this.C = new long[numClasses][numClasses];
        }

        public void reset() {
            for(int i = 0 ; i < C.length ; ++i)
                for(int j = 0 ; j < C.length ; ++j)
                    C[i][j] = 0;
            this.instancesSeen = 0;
        }

        public void update(int iOutput, int jOutput) {
//            System.out.println("iOutput = " + iOutput + " jOutput = " + jOutput);
            this.C[iOutput][jOutput]++;
            this.instancesSeen++;
        }

        public double theta1() {
            double sum = 0.0;
            for(int i = 0 ; i < C.length ; ++i)
                sum += C[i][i];

            return sum / (double) instancesSeen;
        }

        public double theta2() {
            double sum1 = 0.0, sum2 = 0.0, sum = 0.0;

            for(int i = 0 ; i < C.length ; ++i) {
                for(int j = 0 ; j < C.length ; ++j) {
                    sum1 += C[i][j];
                    sum2 += C[j][i];
                }
                //	System.out.println("column = (" + sum1 + "," + (sum1 / (double) instancesSeen) + ") row = (" + sum2 + "," + (sum2 / (double) instancesSeen) + ")");
                sum += (sum1 / (double) instancesSeen) * (sum2 / (double) instancesSeen);
                sum1 = sum2 = 0;
            }
            return sum;
        }

        public int getInstancesSeen() {
            return this.instancesSeen;
        }

        @Override
        public String toString() {
            StringBuilder buffer = new StringBuilder();

            buffer.append("Instances seen = ");
            buffer.append(this.instancesSeen);
            buffer.append(" Theta1 = ");
            buffer.append(this.theta1());
            buffer.append(" Theta2 = ");
            buffer.append(this.theta2());
            buffer.append(" K = ");
            buffer.append(k());
            buffer.append("\n");

            buffer.append('*');
            buffer.append('\t');
            for(int i = 0 ; i < numClasses ; ++i) {
                buffer.append(i);
                buffer.append('\t');
            }
            buffer.append('\n');
            for(int i = 0 ; i < numClasses ; ++i){
                buffer.append(i);
                buffer.append('\t');
                for(int j = 0 ; j < numClasses ; ++j) {
                    buffer.append(C[i][j]);
                    buffer.append('\t');
                }
                buffer.append('\n');
            }
            return buffer.toString();
        }

        public double k() {
            double t1 = theta1(), t2 = theta2();
            return (t1 - t2) / (double) (1.0 - t2);
        }
    }

}
