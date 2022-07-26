package moa.classifiers.semisupervised;

import com.github.javacliparser.*;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.SemiSupervisedLearner;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.meta.StreamingRandomPatches;
import moa.core.*;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.options.ClassOption;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;

public class SLEADE extends AbstractClassifier implements SemiSupervisedLearner, MultiClassClassifier {

    public ClassOption baseEnsembleOption = new ClassOption("baseEnsemble", 'l',
            "Ensemble to train on instances.", StreamingRandomPatches.class, "StreamingRandomPatches");

    public MultiChoiceOption confidenceStrategyOption = new MultiChoiceOption("confidenceStrategy", 'b',
            "Defines the strategy to calculate the confidence. SumVotes = uses votes directly, ArgMax = sets 1 to the argmax of each vote array",
            new String[]{"Sum", "ArgMax"},
            new String[]{"Sum", "ArgMax"}, 1);

//    public FlagOption enableAutoWeightShrinkageOption = new FlagOption("enableAutoWeightShrinkage", 'e',
//            "If set, the weight shrinkage is ignored and this is used instead.");

    public FlagOption enableRandomThresholdOption = new FlagOption("enableRandomThreshold", 'q',
            "If set, the minConfidenceThreshold is ignored and a random value is used.");

    public MultiChoiceOption autoWeightShrinkageOption = new MultiChoiceOption("autoWeightShrinkage", 'e',
            "Different strategies for automatically setting the weight shrinkage (ws) value",
            new String[]{"Constant", "LabeledDivTotal", "LabeledNoWarmupDivTotal"},
            new String[]{"Constant", "LabeledDivTotal", "LabeledNoWarmupDivTotal"}, 2);

    public MultiChoiceOption SSLStrategyOption = new MultiChoiceOption("SSLStrategy", 'p',
            "Defines the strategy to cope with unlabeled instances, i.e. the semi-supervised learning (SSL) strategy.",
            new String[]{"MultiViewMinKappa", "MultiViewMinKappaConfidence"},
            new String[]{"MultiViewMinKappa", "MultiViewMinKappaConfidence"}, 1);

    public FloatOption SSL_minConfidenceOption = new FloatOption("SSL_minConfidence", 'm',
            "Minimum confidence to use unlabeled instance to train.", 0.00, 0.0, 1.0);

    public MultiChoiceOption weightFunctionOption = new MultiChoiceOption("weightFunction", 'w',
            "Defines the function to weight pseudo-labeled instances. ",
            new String[]{"Constant1", "Confidence", "ConfidenceWeightShrinkage", "UnsupervisedDetectionWeightShrinkage"},
            new String[]{"Constant1", "Confidence", "ConfidenceWeightShrinkage", "UnsupervisedDetectionWeightShrinkage"}, 2);

    public MultiChoiceOption pairingFunctionOption = new MultiChoiceOption("pairingFunction", 't',
            "Defines the function to pair learners, default is minimum Kappa",
            new String[]{"MinKappa", "Random", "MajorityTrainsMinority"},
            new String[]{"MinKappa", "Random", "MajorityTrainsMinority"}, 2);

    public FloatOption SSL_weightShrinkageOption = new FloatOption("SSL_weightShrinkageOption", 'n',
            "The pseudo-labeled instances will be weighted according to instance weight * 1/WS.", 100.0, 0.0, Integer.MAX_VALUE);

//    public FlagOption generateKappaNetwork = new FlagOption("generateKappaNetwork", 'b',
//            "Whether to generate the Kappa network");

    public FlagOption pseudoLabelChangeDetectionUpdateAllowedOption = new FlagOption("pseudoLabelChangeDetectionUpdateAllowed", 'c',
            "Allow pseudo-labeled instances to be used to update the change detector");


    public FlagOption useUnsupervisedDriftDetectionOption = new FlagOption("useUnsupervisedDriftDetection", 's',
            "Whether or not to use the unsupervised drift detection reaction strategy.");


    public ClassOption studentLearnerForUnsupervisedDriftDetectionOption = new ClassOption("studentLearnerForUnsupervisedDriftDetection", 'g',
            "Student to mimic the ensemble. It is used for unsupervised drift detection.", Classifier.class, "trees.HoeffdingTree -g 50 -c 0.01");

    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-5");

//    public ClassOption warningDetectionMethodOption = new ClassOption("warningDetectionMethod", 'k',
//            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-3");


    public IntOption unsupervisedDetectionWeightWindowOption = new IntOption("unsupervisedDetectionWeightWindow", 'z',
            "The labeledInstancesBuffer length of the sigmoid functions for the unsupervised drift detection pseudo-labeling weighting.", 20, 0, Integer.MAX_VALUE);

    public IntOption labeledWindowLimitOption = new IntOption( "labeledWindowLimit", 'j', "The maximum number of instances to store", 100, 1, Integer.MAX_VALUE);


    public FlagOption debugShowTTTAccuracyConfidenceOption = new FlagOption("debugShowTTTAccuracyConfidence", 'v',
            "Writes to the output after every <debugFrequency> instances the accuracy " +
                    "and avg confidence in incorrect predictions and avg confidence in correct predictions");

    public FileOption debugOutputFileOption = new FileOption("debugOutputFile", 'o',
            "File to append debug information.", null, "csv", true);


    public FileOption debugOutputConfidencePredictionsFileOption = new FileOption(
            "debugOutputConfidencePredictionsFile", 'f',
            "File to append confidences and predictions information.", null, "csv", true);


    public IntOption debugFrequencyOption = new IntOption("debugFrequency", 'i',
            "The labeledInstancesBuffer length of the sigmoid functions for the unsupervised drift detection pseudo-labeling weighting.", 1000, 0, Integer.MAX_VALUE);

    public FlagOption debugPerfectPseudoLabelingOption = new FlagOption("debugPerfectPseudoLabeling", 'd',
            "CAUTION: THIS OVERRIDES THE PSEUDO-LABELING STRATEGY AND ALWAYS CHOOSE THE CORRECT LABEL AS THE PSEUDO-LABEL, FOR DEBUG ONLY!");

    public StringOption gtDriftLocationOption = new StringOption("gtDriftLocation", 'a',
            "If set, it is used to calculate drift detection accuracy. format: 'dd_location1,dd_location2,...,dd_locationX", "");

//    public IntOption numberOfSeedClassifiers = new IntOption("numberOfSeedClassifiers", 'd',
//            "The number of models to consider as seeds for subnetwork formation.", 2, 1, Integer.MAX_VALUE);

    StreamingRandomPatches baseEnsemble;

    Classifier student;

    protected ChangeDetector driftDetectionMethod;
//    protected ChangeDetector warningDetectionMethod;
//    protected boolean warningPeriodOn;


    protected long lastDriftDetectedAt;
    // For debug only.
    protected long lastDriftDetectedAtUnlabelled;
    private WindowClassificationPerformanceEvaluator evaluatorSupervisedDebug;

    private WindowClassificationPerformanceEvaluator[] evaluatorEnsembleMembersDebug;

    // USED FOR DEBUGGING THE TTT ACCURACY OF THE ENSEMBLE AT ANY TIME, ASSUMES ACCESS TO THE LABELED DATA
    private BasicClassificationPerformanceEvaluator evaluatorBasicDebugEnsemble;

    protected C[][] pairsOutputsKappa;
//    protected ArrayList<Integer> seedsKappa;
//    protected int[] subnetworksKappa;

    // Buffer of labeled instances
    protected Instances labeledInstancesBuffer;

    // This is only called during supervised training, thus it is a fair estimation of the predictive performance
    private BasicClassificationPerformanceEvaluator[] evaluatorMembers;

    // Statistics
    protected long instancesSeen; // count only labeled (updated only on trainOnInstance()
    protected long allInstancesSeen; // Unlabeled and labeled
    protected long instancesPseudoLabeled;
    protected long instancesCorrectPseudoLabeled;
    protected double currentAverageKappa;
    protected double currentMinKappa;
    protected double currentMaxKappa;
    protected int numberOfUnsupervisedDriftsDetected;
//    protected int numberOfUnsupervisedWarningsDetected;

    // To count the number of warmup period instances
    protected long initialWarmupTrainingCounter;

//    autoWeightShrinkageOption
    public static final int AUTO_WEIGHT_SHRINKAGE_CONSTANT = 0; // use the default constant value from SSL_weightShrinkageOption
    public static final int AUTO_WEIGHT_SHRINKAGE_LABELED_DIV_TOTAL = 1;
    public static final int AUTO_WEIGHT_SHRINKAGE_LABELED_IGNORE_WARMUP_DIV_TOTAL = 2;

    public static final int SSL_MULTI_VIEW_MIN_KAPPA = 0;
    public static final int SSL_MULTI_VIEW_MIN_KAPPA_CONFIDENCE = 1;

    public static final int SSL_WEIGHT_CONSTANT_1 = 0;
    public static final int SSL_WEIGHT_CONFIDENCE = 1;
    public static final int SSL_WEIGHT_CONFIDENCE_WS = 2;
    public static final int SSL_WEIGHT_UNSUPERVISED_DETECTION_WEIGHT_SHRINKAGE = 3;   // UnsupervisedDetectionWeightShrinkage

    public static final int SSL_PAIRING_MIN_KAPPA = 0;
    public static final int SSL_PAIRING_RANDOM = 1;
    public static final int SSL_PAIRING_MAJORITY_TRAINS_MINORITY = 2;

    public static final int SSL_CONFIDENCE_STRATEGY_SUM = 0;
    public static final int SSL_CONFIDENCE_STRATEGY_ARGMAX = 1;

    PrintStream outputDebugStream = null;
    PrintStream outputConfidencePredictionDebugStream = null;

    int[] gtDriftLocations = null;

    @Override
    public void setModelContext(InstancesHeader context) {
        try {
            this.labeledInstancesBuffer = new Instances(context,0); //new StringReader(context.toString())
            this.labeledInstancesBuffer.setClassIndex(context.classIndex());
        } catch(Exception e) {
            System.err.println("Error: no Model Context available.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    protected void reactToDrift(Instance instance) {
//        System.out.println("reactToDrift:");
        double sum = 0.0, mean = 0.0;
        int minIndex = 0, maxIndex = 0;

        for(int i = 0; i < this.evaluatorMembers.length ; ++i) {
            double acc = this.evaluatorMembers[i].getPerformanceMeasurements()[1].getValue();

            if(this.evaluatorMembers[minIndex].getPerformanceMeasurements()[1].getValue() > acc)
                minIndex = i;
            if(this.evaluatorMembers[maxIndex].getPerformanceMeasurements()[1].getValue() <= acc)
                maxIndex = i;

            sum += acc;
//            System.out.println("\tLearner (" + i + ") = " + acc);
        }
        mean = sum / this.evaluatorMembers.length;

//        System.out.println("mean: " + mean +
//                " min: " + this.evaluatorMembers[minIndex].getPerformanceMeasurements()[1].getValue() +
//                " max: " + this.evaluatorMembers[maxIndex].getPerformanceMeasurements()[1].getValue());
//        for(int i = 0 ; i < accuracies.numValues() ; ++i)
//            System.out.println(accuracies.getValue(i));

        StreamingRandomPatches.StreamingRandomPatchesClassifier[] ensemble = this.baseEnsemble.getEnsembleMembers();
        ArrayList<Integer> resetIndexes = new ArrayList<>();

        for(int i = 0 ; i < ensemble.length ; ++i) {
            if (this.evaluatorMembers[i].getPerformanceMeasurements()[1].getValue() < mean) {
//                System.out.println("Resetting learner(" + i + ")");
                ensemble[i].reset(instance, this.instancesSeen, this.classifierRandom);
                resetIndexes.add(i);
            }
        }

        // Train the newly added learners on the buffered labeled instances
        for(int i = 0 ; i < resetIndexes.size() ; ++i) {
            for(int j = 0 ; j < this.labeledInstancesBuffer.numInstances() ; ++j) {
                ensemble[resetIndexes.get(i)].trainOnInstance(this.labeledInstancesBuffer.get(j), 1, this.instancesSeen, this.classifierRandom, false);
            }
        }

    }

    private boolean trainAndUpdateStudentDetector(Instance instance) {
        double[] ensembleVotes = this.baseEnsemble.getVotesForInstance(instance);
        int ensemblePrediction = Utils.maxIndex(ensembleVotes);

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
            if(this.outputDebugStream != null)
                this.outputDebugStream.println(this.allInstancesSeen + ", " + this.instancesSeen + ", "+ this.evaluatorSupervisedDebug.getPerformanceMeasurements()[1].getValue() +", SLEAD");

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
        this.baseEnsemble = (StreamingRandomPatches) getPreparedClassOption(this.baseEnsembleOption);
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

        this.initialWarmupTrainingCounter = 0;

        this.numberOfUnsupervisedDriftsDetected = 0;

        this.evaluatorSupervisedDebug = new WindowClassificationPerformanceEvaluator();
        this.evaluatorSupervisedDebug.widthOption.setValue(this.debugFrequencyOption.getValue());
        this.evaluatorSupervisedDebug.reset();

        this.evaluatorBasicDebugEnsemble = new BasicClassificationPerformanceEvaluator();
        this.evaluatorBasicDebugEnsemble.reset();

        // Drift detector ground-truth
        String ddStr = this.gtDriftLocationOption.getValue();
        if(ddStr.length() != 0) {
            String[] ddStrArr = ddStr.split(",");
            this.gtDriftLocations = new int[ddStrArr.length];
            for(int i = 0 ; i < ddStrArr.length ; ++i)
                this.gtDriftLocations[i] = Integer.parseInt(ddStrArr[i]);
        }

        // File for debug
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

        // File for debug
        File outputConfidencePredictionsDebugFile = this.debugOutputConfidencePredictionsFileOption.getFile();
        if (outputConfidencePredictionsDebugFile != null) {
            try {
                if (outputConfidencePredictionsDebugFile.exists()) {
                    outputConfidencePredictionDebugStream = new PrintStream(
                            new FileOutputStream(outputConfidencePredictionsDebugFile, true), true);
                } else {
                    outputConfidencePredictionDebugStream = new PrintStream(
                            new FileOutputStream(outputConfidencePredictionsDebugFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open confidence prediction debug file: " + outputConfidencePredictionsDebugFile, ex);
            }
        }

    }

    private void initDebugMode() {

        if(this.evaluatorEnsembleMembersDebug == null)
            // This is for debug purposes only as it is invoked during getVotesForInstance
            this.evaluatorEnsembleMembersDebug = new WindowClassificationPerformanceEvaluator[this.baseEnsemble.getEnsembleMembers().length];

        // Gambi!
        if(this.outputDebugStream != null) {
            for (int i = 0; i < this.baseEnsemble.getEnsembleMembers().length; ++i) {
                // nao precisa fazer isso! Os membros do ensemble nao precisam escrever na saida de debug...
//                this.baseEnsemble.getEnsembleMembers()[i].outputDebugStream = this.outputDebugStream;
                this.evaluatorEnsembleMembersDebug[i] = new WindowClassificationPerformanceEvaluator();
                this.evaluatorEnsembleMembersDebug[i].widthOption.setValue(this.debugFrequencyOption.getValue());
            }

            // Even more Gambi!
            this.outputDebugStream.print("#instances-labeled,#instances,#labeled_instances,windowed_accuracy(w=" + this.debugFrequencyOption.getValue() + ")," + "drift_descriptor,");

            for (int i = 0; i < this.baseEnsemble.getEnsembleMembers().length; ++i) {
                this.outputDebugStream.print("TTT_acc(" + this.baseEnsemble.getEnsembleMembers()[i].indexOriginal + "),");
            }

            for (int i = 0; i < this.baseEnsemble.getEnsembleMembers().length; ++i) {
                this.outputDebugStream.print("win_acc(" + this.baseEnsemble.getEnsembleMembers()[i].indexOriginal + "),");
            }

            for (int i = 0; i < this.baseEnsemble.getEnsembleMembers().length; ++i) {
                this.outputDebugStream.print("#lab_inst("
                        + this.baseEnsemble.getEnsembleMembers()[i].indexOriginal + "),");
            }
            this.outputDebugStream.println();
        }

        // Mais gambi, agora pra inicializar o confidence prediction debug stream
        if(outputConfidencePredictionDebugStream != null) {
            StringBuilder stbPred = new StringBuilder();
            StringBuilder stbConf = new StringBuilder();
            StringBuilder stbTTTacc = new StringBuilder();
            StringBuilder stbWINacc = new StringBuilder();
            StringBuilder stbComplementPred = new StringBuilder();
            StringBuilder stbComplementConf = new StringBuilder();
            StringBuilder stbWeight = new StringBuilder();

            StringBuilder stbUsedPseudo = new StringBuilder();
            for(int i = 0 ; i < this.baseEnsemble.ensembleSizeOption.getValue() ; ++i) {
                stbPred.append("Pred(" + i + "),");
                stbConf.append("Conf(" + i + "),");
                stbComplementPred.append("Pred(L\\" + i + "),");
                stbComplementConf.append("Conf(L\\" + i + "),");
                stbUsedPseudo.append("Pseudo(" + i + "),");



                stbTTTacc.append("TTT_acc(" + i + "),");
                stbWINacc.append("WIN_acc(" + i + "),");
                stbWeight.append("weightPse(" + i + "),");
            }

            // Predicao do ensemble, predicao de cada membro
            this.outputConfidencePredictionDebugStream.print("#instance,groundTruthLabel,Ensemble_Pred,Ensemble_Conf,Ensemble_TTT_acc,Ensemble_WIN_acc,");
            this.outputConfidencePredictionDebugStream.print(stbPred.toString());
            this.outputConfidencePredictionDebugStream.print(stbConf.toString());
            this.outputConfidencePredictionDebugStream.print(stbComplementPred.toString());
            this.outputConfidencePredictionDebugStream.print(stbComplementConf.toString());
            this.outputConfidencePredictionDebugStream.print(stbUsedPseudo.toString());
            this.outputConfidencePredictionDebugStream.print(stbTTTacc.toString());
            this.outputConfidencePredictionDebugStream.print(stbWINacc.toString());
            this.outputConfidencePredictionDebugStream.print(stbWeight.toString());

            this.outputConfidencePredictionDebugStream.println();
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        ++this.instancesSeen;
        ++this.allInstancesSeen;

//        estimateSupervisedLearningAccuracy(instance);

        if (this.baseEnsemble.getEnsembleMembers() == null) {
            this.baseEnsemble.initEnsemble(instance);

            initDebugMode();

            this.evaluatorMembers = new BasicClassificationPerformanceEvaluator[this.baseEnsemble.getEnsembleMembers().length];

            for (int i = 0; i < this.baseEnsemble.getEnsembleMembers().length; ++i)
                this.evaluatorMembers[i] = new BasicClassificationPerformanceEvaluator();
        }


        if(this.baseEnsemble != null && this.baseEnsemble.getEnsembleMembers() != null) {
            if(this.evaluatorMembers != null) {
                InstanceExample example = new InstanceExample(instance);
                for (int i = 0; i < this.baseEnsemble.getEnsembleMembers().length; ++i) {
                    double[] voteEnsembleMember = this.baseEnsemble.getEnsembleMembers()[i].getVotesForInstance(instance);
                    this.evaluatorMembers[i].addResult(example, voteEnsembleMember);
                }
            }
            else {
                // This is not for debug! This one is only updated during supervised learning.
                this.evaluatorMembers = new BasicClassificationPerformanceEvaluator[this.baseEnsemble.getEnsembleMembers().length];

                for (int i = 0; i < this.baseEnsemble.getEnsembleMembers().length; ++i)
                    this.evaluatorMembers[i] = new BasicClassificationPerformanceEvaluator();
            }
        }

        // **** DEBUG CODE **** //
//        if(/*this.instancesSeen >= 1000 && */ this.instancesSeen == 1970) { // && this.instancesSeen <= 2000) {
//            System.out.println("x, prob_from_drift_point (1000), prob_from_instancesSeen_point (1600), min(pfdp, pfisp)");
//            for(int x = 500 ; x <= 1800 ; x+=5) {
//                double prob_drift = sigmoid(x, 1000, 20);
//                double prob_current = 1 - sigmoid(x, 1600-20, 20);
//
//                System.out.println(/*1000 + ", " + */x + ", "
//                        + prob_drift + "," +
//                        prob_current + "," +
//                        min(prob_drift, prob_current) + "," +
//                        weightAge(x, 1000, 1600/*this.lastDriftDetectedAt, this.instancesSeen*/, 20));
//            }
//        }
        // **** DEBUG CODE **** //

        // TODO: add methods to access base models for other ensembles
        StreamingRandomPatches.StreamingRandomPatchesClassifier[] ensembleMembers = this.baseEnsemble.getEnsembleMembers();
//        Classifier[] ensembleMembers = this.baseEnsemble.getSubClassifiers();

        int [][] changeStatus = kappaUpdateFirst(instance, ensembleMembers);

        // Store the predictions of each learner.
        HashMap<Integer, Integer> classifiersExactPredictions = classifiersPredictions(instance, ensembleMembers);

        /********* TRAIN THE ENSEMBLE *********/
        this.baseEnsemble.trainOnInstanceImpl(instance);
        /********* TRAIN THE ENSEMBLE *********/

        if(useUnsupervisedDriftDetectionOption.isSet()) {
            if (this.labeledInstancesBuffer == null) {
                this.labeledInstancesBuffer = new Instances(instance.dataset());
            }
            if (this.labeledWindowLimitOption.getValue() <= this.labeledInstancesBuffer.numInstances()) {
                this.labeledInstancesBuffer.delete(0);
            }
            this.labeledInstancesBuffer.add(instance);


            boolean driftDetected = trainAndUpdateStudentDetector(instance);
            if (driftDetected)
                reactToDrift(instance);
        }

        kappaUpdateSecond(changeStatus, ensembleMembers, classifiersExactPredictions);
    }

    @Override
    public void addInitialWarmupTrainingInstances() {
        this.initialWarmupTrainingCounter++;
    }

    @Override
    public int trainOnUnlabeledInstance(Instance instance) {
        this.allInstancesSeen++;

//        estimateSupervisedLearningAccuracy(instance);

        if(debugPerfectPseudoLabelingOption.isSet() && instance.classIsMissing()) {
            throw new RuntimeException("Pseudo-labeling debug is on, but the true class is not available");
        }
        // DEBUG confidence and accuracy
        int groundTruthLabel = -1;
        int ensemblePred = -1;
        double ensembleConf = -1.0;
        double[] learnersPred = new double[this.baseEnsemble.ensembleSizeOption.getValue()];
        double[] learnersConf = new double[this.baseEnsemble.ensembleSizeOption.getValue()];
        double[] complementPred = new double[this.baseEnsemble.ensembleSizeOption.getValue()];
        double[] complementConf = new double[this.baseEnsemble.ensembleSizeOption.getValue()];
        int[] pseudoUsed = new int[this.baseEnsemble.ensembleSizeOption.getValue()];
        double[] learnerTTT_acc = new double[this.baseEnsemble.ensembleSizeOption.getValue()];
        double[] learnerWIN_acc = new double[this.baseEnsemble.ensembleSizeOption.getValue()];
        double[] learnersWeight = new double[this.baseEnsemble.ensembleSizeOption.getValue()];

        int predictedIndex = -1;

        // TODO: add methods to access base models for other ensembles
        StreamingRandomPatches.StreamingRandomPatchesClassifier[] ensembleMembers = this.baseEnsemble.getEnsembleMembers();

        int [][] changeStatus = kappaUpdateFirst(instance, ensembleMembers);

////         Store the predictions of each learner.
        HashMap<Integer, Integer> classifiersExactPredictions = classifiersPredictions(instance, ensembleMembers);

        boolean changeDetectionUpdateAllowed = this.pseudoLabelChangeDetectionUpdateAllowedOption.isSet();

        if(this.pairsOutputsKappa == null)
            throw new RuntimeException("No kappa network for MULTI_VIEW_MIN_KAPPA_CONFIDENCE ");

        ArrayList<Integer> indicesNoReposition = new ArrayList<>();
        for(int i = 0 ; i < ensembleMembers.length ; ++i)
            indicesNoReposition.add(i);

        double[] votesRawOthers = new double[instance.numClasses()];
        double minIdxEstimatedAccuracy = -1.0;
        double confidenceForMajMin = -1;

        for(int i = 0 ; i < ensembleMembers.length ; ++i) {

            int idxMin = -1;
            switch(this.pairingFunctionOption.getChosenIndex()) {
                case SSL_PAIRING_MIN_KAPPA:
                    for (int j = 0; j < ensembleMembers.length; ++j) {
                        if (j != i) {
                            // Find the smallest kappa, i.e., far from
                            double currentKappa = this.getKappa(i, j);
                            if (idxMin == -1 || currentKappa < this.getKappa(i, idxMin))
                                idxMin = j;
                        }
                    }
                    votesRawOthers = ensembleMembers[idxMin].getVotesForInstance(instance);
                    minIdxEstimatedAccuracy = ensembleMembers[idxMin].evaluator.getPerformanceMeasurements()[1].getValue() / 100.0;
                    break;
                case SSL_PAIRING_RANDOM:
                    // keep randomly choosing until idxMinKappa != i
                    do {
                        idxMin = Math.abs(this.classifierRandom.nextInt()) % ensembleMembers.length;
                    } while(idxMin == i);
                    votesRawOthers = ensembleMembers[idxMin].getVotesForInstance(instance);
                    minIdxEstimatedAccuracy = ensembleMembers[idxMin].evaluator.getPerformanceMeasurements()[1].getValue() / 100.0;
                    break;
                case SSL_PAIRING_MAJORITY_TRAINS_MINORITY:
                    // TODO: Extremely naive approach, getting the same predictions (Ensemble size - 1) times.
                    // Could create an array with predictions the first time and lazily select which learners are used.
                    // For the future! Make it faster once it is right!
                    // Versao antiga!
//                    votesRawOthers = getVotesForComplementSubEnsemble(instance, i, this.baseEnsemble.getEnsembleMembers());
                    // Assumption: the complement ensemble will not yield a random prediction! Thus we can assume a performance of 0.5
                    // We could also estimate it by summing all the performance measurements for it and dividing by the ensemble size

                    /****** OLD strategy for the confidence ******/
                    if(confidenceStrategyOption.getChosenIndex() == SSL_CONFIDENCE_STRATEGY_SUM) {
                        votesRawOthers = getVotesForComplementSubEnsemble(instance, i, this.baseEnsemble.getEnsembleMembers());

                    } else { /****** NEW strategy for the confidence ******/
                        votesRawOthers = getExactVotesForComplementSubEnsemble(instance, i, this.baseEnsemble.getEnsembleMembers());
                    }
                    DoubleVector votes = new DoubleVector(votesRawOthers);
                    int predictedIndexByVotes = votes.maxIndex();
                    confidenceForMajMin = votes.getValue(predictedIndexByVotes) / votes.sumOfValues();

                    minIdxEstimatedAccuracy = 0.5;//ensembleMembers[idxMinKappa].evaluator.getPerformanceMeasurements()[1].getValue() / 100.0;
                    break;
            }

//            if(this.pairingFunctionOption.getChosenIndex() != SSL_PAIRING_MAJORITY_TRAINS_MINORITY) ;

            DoubleVector votesMinKappa = new DoubleVector(votesRawOthers);
            int predictedIndexByMinKappa = votesMinKappa.maxIndex();

            Instance instanceWithPredictedLabelByMinKappa = instance.copy();

            /*** DEBUG PURPOSES ONLY, ON REAL TESTS, THIS SHOULD NEVER BE USED ***/
            if (this.debugPerfectPseudoLabelingOption.isSet())
                instanceWithPredictedLabelByMinKappa.setClassValue(instance.classValue());
            else
                instanceWithPredictedLabelByMinKappa.setClassValue(predictedIndexByMinKappa);

            // Sanity check...
            if (predictedIndexByMinKappa >= 0 && votesMinKappa.sumOfValues() > 0.0 && minIdxEstimatedAccuracy > 0.0) {

                double weightMinKappa = -1;

                double weightShrinkage = 1 / this.SSL_weightShrinkageOption.getValue();

                switch(autoWeightShrinkageOption.getChosenIndex()) {
                    case AUTO_WEIGHT_SHRINKAGE_CONSTANT:
                        weightShrinkage = 1 / this.SSL_weightShrinkageOption.getValue();
                        break;
                    case AUTO_WEIGHT_SHRINKAGE_LABELED_DIV_TOTAL:
                        weightShrinkage = this.instancesSeen / (this.allInstancesSeen + 0.0000001);
                        break;

                    case AUTO_WEIGHT_SHRINKAGE_LABELED_IGNORE_WARMUP_DIV_TOTAL:
                        weightShrinkage = (this.instancesSeen - this.initialWarmupTrainingCounter) / (this.allInstancesSeen - this.initialWarmupTrainingCounter + 0.0000001);
                        break;
                }

                switch(weightFunctionOption.getChosenIndex()) {
                    case SSL_WEIGHT_CONSTANT_1:
                        weightMinKappa = 1;
                        break;
                    case SSL_WEIGHT_CONFIDENCE:
                        weightMinKappa = votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues();
                        break;
                    case SSL_WEIGHT_CONFIDENCE_WS:
                        weightMinKappa = votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues()
                                * weightShrinkage;
                        break;

                    case SSL_WEIGHT_UNSUPERVISED_DETECTION_WEIGHT_SHRINKAGE:
                        // weight = confidence on prediction * weight_age()
//                        weightMinKappa = votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues()
//                                / ensembleMembers[i].createdOn;

                        weightMinKappa = votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues()
                                * weightAge(ensembleMembers[idxMin].createdOn, this.lastDriftDetectedAt, this.instancesSeen,
                                this.unsupervisedDetectionWeightWindowOption.getValue());
                        break;
                }

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
                        // if majority trains minority
//                        double confidenceByMinKappa = -1; // votesMinKappa.getValue(predictedIndexByMinKappa) / votesMinKappa.sumOfValues();

                        DoubleVector votesByCurrent = new DoubleVector(ensembleMembers[i].getVotesForInstance(instance));
                        int predictedByCurrent = votesByCurrent.maxIndex();
                        double confidenceByCurrent = votesByCurrent.getValue(predictedByCurrent) / votesByCurrent.sumOfValues();

                        double minConfidenceThreshold = this.SSL_minConfidenceOption.getValue();

                        // NEW: this overwrite the minimum confidence threshold set by the user and apply a random one.
                        if(this.enableRandomThresholdOption.isSet()) {
                            minConfidenceThreshold = classifierRandom.nextDouble();
                        }

                        if (confidenceForMajMin > minConfidenceThreshold &&
                                predictedByCurrent != predictedIndexByMinKappa &&
                                confidenceForMajMin > confidenceByCurrent) {


                            /***** TRAIN MEMBER i WITH PSEUDO-LABELED INSTANCE *****/
                            ensembleMembers[i].trainOnInstance(
                                    instanceWithPredictedLabelByMinKappa, weightMinKappa, this.instancesSeen, this.classifierRandom, changeDetectionUpdateAllowed);

                            if (!instance.classIsMissing() && ((int) instanceWithPredictedLabelByMinKappa.classValue()) == ((int) instance.classValue())) {
                                ++this.instancesCorrectPseudoLabeled;
                                // DEBUG
                                pseudoUsed[i] = 1;
                            }
                            else {
                                // DEBUG
                                pseudoUsed[i] = -1;
                            }
                            ++this.instancesPseudoLabeled;



                        }

                        // DEBUG
                        if(this.outputConfidencePredictionDebugStream != null) {

                            double[] votes = this.baseEnsemble.getVotesForInstance(instance);
                            double sumVotes = Utils.sum(votes);

                            // the weight after considering the confidence as well
//                            groundTruthLabel = (int) instance.classValue();
                            ensemblePred = Utils.maxIndex(votes);
                            ensembleConf = votes[ensemblePred] / sumVotes;

                            learnersWeight[i] = weightMinKappa;
                            learnersPred[i] = predictedByCurrent;
                            learnersConf[i] = confidenceByCurrent;
                            complementPred[i] = predictedIndexByMinKappa;
                            complementConf[i] = confidenceForMajMin;
                            learnerTTT_acc[i] = this.baseEnsemble.getEnsembleMembers()[i].evaluator.getPerformanceMeasurements()[1].getValue();
                            learnerWIN_acc[i] = this.evaluatorEnsembleMembersDebug[i].getPerformanceMeasurements()[1].getValue();
                        }
                        break;
                }
            }
        }

        // Check if changes occurred.
        kappaUpdateSecond(changeStatus, ensembleMembers, classifiersExactPredictions);

        if(useUnsupervisedDriftDetectionOption.isSet()) {
            boolean driftDetected = trainAndUpdateStudentDetector(instance);
            if (driftDetected)
                reactToDrift(instance);
        }

        // Debug accuracy confidence
        if(this.outputConfidencePredictionDebugStream != null) {
            groundTruthLabel = (int) instance.classValue();
            printConfidenceAccuracyDebug(groundTruthLabel, ensemblePred,
                    ensembleConf,
                    learnersPred, learnersConf, complementPred, complementConf,
                    pseudoUsed, learnerTTT_acc, learnerWIN_acc, learnersWeight);
        }
//        else {
//            System.out.println("this.outputConfidencePredictionDebugStream == null");
//        }
        return predictedIndex;
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        if (this.baseEnsemble.getEnsembleMembers() == null) {
            this.baseEnsemble.initEnsemble(instance);
            initDebugMode();
        }

        if(this.evaluatorEnsembleMembersDebug != null && this.evaluatorEnsembleMembersDebug[0] != null)
            estimateLearningAccuracy(instance);
        calculateDriftDetectionMetrics();

        double[] votes = this.baseEnsemble.getVotesForInstance(instance);

        if(this.outputConfidencePredictionDebugStream != null) {
            InstanceExample example = new InstanceExample(instance);
            this.evaluatorBasicDebugEnsemble.addResult(example, votes);
        }
        return votes;
    }


    private void printConfidenceAccuracyDebug(int groundTruthLabel, int ensemblePred, double ensembleConf,
                                              double[] learnersPred, double[] learnersConf,
                                              double[] complementPred, double[] complementConf,
                                              int[] pseudoUsed, double[] learnerTTT_acc,
                                              double[] learnerWIN_acc, double[] learnersWeight) {
        // DEBUG
        if(this.outputConfidencePredictionDebugStream != null) {
// #instance Ensemble_TTT_acc Ensemble_WIN_acc Pred(0)	Conf(0)	Pred(L\0) Conf(L\0)	Pseudo(0) TTT_acc(0) WIN_acc(0)

            // #instance
            outputConfidencePredictionDebugStream.print(this.allInstancesSeen + ",");

            outputConfidencePredictionDebugStream.print(groundTruthLabel + ",");

            outputConfidencePredictionDebugStream.print(ensemblePred + ",");

            // Confidence
            outputConfidencePredictionDebugStream.print(ensembleConf + ",");

            // Ensemble_TTT_acc
            outputConfidencePredictionDebugStream.print(
                    this.evaluatorBasicDebugEnsemble.getPerformanceMeasurements()[1].getValue() + ",");

            outputConfidencePredictionDebugStream.print(
                    this.evaluatorSupervisedDebug.getPerformanceMeasurements()[1].getValue() + ",");

            for(int i = 0 ; i < learnersPred.length ; ++i) {
                this.outputConfidencePredictionDebugStream.print(learnersPred[i] + ",");
            }

            for(int i = 0 ; i < learnersConf.length ; ++i) {
                this.outputConfidencePredictionDebugStream.print(learnersConf[i] + ",");
            }

            for(int i = 0 ; i < complementPred.length ; ++i) {
                this.outputConfidencePredictionDebugStream.print(complementPred[i] + ",");
            }

            for(int i = 0 ; i < complementConf.length ; ++i) {
                this.outputConfidencePredictionDebugStream.print(complementConf[i] + ",");
            }

            for(int i = 0 ; i < pseudoUsed.length ; ++i) {
                this.outputConfidencePredictionDebugStream.print(pseudoUsed[i] + ",");
            }

            for(int i = 0 ; i < learnerTTT_acc.length ; ++i) {
                this.outputConfidencePredictionDebugStream.print(learnerTTT_acc[i] + ",");
            }

            for(int i = 0 ; i < learnerWIN_acc.length ; ++i) {
                this.outputConfidencePredictionDebugStream.print(learnerWIN_acc[i] + ",");
            }

            for(int i = 0 ; i < learnersWeight.length ; ++i) {
                this.outputConfidencePredictionDebugStream.print(learnersWeight[i] + ",");
            }

//            for(int i = 0 ; i < this.baseEnsemble.getEnsembleMembers().length ; ++i) {
//                this.outputDebugStream.print(this.evaluatorEnsembleMembersDebug[i].getPerformanceMeasurements()[1].getValue() + ",");
//            }
//
//            for(int i = 0 ; i < this.baseEnsemble.getEnsembleMembers().length ; ++i) {
//                this.outputDebugStream.print((this.instancesSeen - this.baseEnsemble.getEnsembleMembers()[i].createdOn) + ",");
//            }
            this.outputConfidencePredictionDebugStream.println();
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        // instances seen * the number of ensemble members
//        return new Measurement[]{new Measurement("pseudo labeled",
//                this.instancesPseudoLabeled / (double) (this.instancesSeen * this.baseEnsemble.ensembleSizeOption.getValue()) * 100),
//                new Measurement("correct pseudo labeled",
//                        this.instancesPseudoLabeled != 0 ? this.instancesCorrectPseudoLabeled / (double) this.instancesPseudoLabeled * 100 : 0)
//        };

        int totalDriftResets = 0;
        for(int i = 0 ; i < this.baseEnsemble.getEnsembleMembers().length ; ++i)
            totalDriftResets += this.baseEnsemble.getEnsembleMembers()[i].numberOfDriftsDetected;

        // DEBUG
        if(this.debugShowTTTAccuracyConfidenceOption.isSet() && this.outputDebugStream != null) {
//            if(this.allInstancesSeen % this.debugFrequencyOption.getValue() == 0)
            outputDebugStream.print((this.allInstancesSeen-this.instancesSeen) + "," + this.allInstancesSeen + "," + this.instancesSeen + "," +
                    this.evaluatorSupervisedDebug.getPerformanceMeasurements()[1].getValue() + ",,"); //+ ", avg conf incorr ???, avg conf corr ???");

            for(int i = 0 ; i < this.baseEnsemble.getEnsembleMembers().length ; ++i) {
                this.outputDebugStream.print(this.baseEnsemble.getEnsembleMembers()[i].evaluator.getPerformanceMeasurements()[1].getValue() + ",");
            }

            for(int i = 0 ; i < this.baseEnsemble.getEnsembleMembers().length ; ++i) {
                this.outputDebugStream.print(this.evaluatorEnsembleMembersDebug[i].getPerformanceMeasurements()[1].getValue() + ",");
            }

            for(int i = 0 ; i < this.baseEnsemble.getEnsembleMembers().length ; ++i) {
                this.outputDebugStream.print((this.instancesSeen - this.baseEnsemble.getEnsembleMembers()[i].createdOn) + ",");
            }
            this.outputDebugStream.println();
        }
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
//                new Measurement("accuracy supervised learner", this.evaluatorSupervisedDebug.getPerformanceMeasurements()[1].getValue())
        };
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return true;
    }


    private double[] getVotesForComplementSubEnsemble(Instance instance, int idxLearner, StreamingRandomPatches.StreamingRandomPatchesClassifier[] ensembleMembers) {
        // Store the predictions of each learner.
        DoubleVector votes = new DoubleVector();

        for (int i = 0; i < ensembleMembers.length; i++) {
            if(i != idxLearner) {
                double[] rawVote = ensembleMembers[i].getVotesForInstance(instance);
                DoubleVector vote = new DoubleVector(rawVote);
                votes.addValues(vote);
            }

//          TODO: This is internal to SRP, when we move to a generic approach, we may need to do that in here as well, but with a dedicated evaluator.
//          ensembleMembers[i].evaluator.addResult(example, vote.getArrayRef());
        }
        return votes.getArrayCopy();
    }


    private double[] getExactVotesForComplementSubEnsemble(Instance instance, int idxLearner,
                                                           StreamingRandomPatches.StreamingRandomPatchesClassifier[] ensembleMembers) {
        // Store the predictions of each learner.
        DoubleVector votes = new DoubleVector();


        for (int i = 0; i < ensembleMembers.length; i++) {
            if(i != idxLearner) {
                double[] rawVote = ensembleMembers[i].getVotesForInstance(instance);
                DoubleVector vote = new DoubleVector(rawVote);
                int maxIndex = vote.maxIndex();
                double[] dummyVote = new double[instance.numClasses()];
                dummyVote[maxIndex] = 1;
                if(vote.getValue(maxIndex) > 0)
                    votes.addValues(dummyVote);
                else
                    votes.addValues(vote); // add the array with all 0's
            }

//          TODO: This is internal to SRP, when we move to a generic approach, we may need to do that in here as well, but with a dedicated evaluator.
//          ensembleMembers[i].evaluator.addResult(example, vote.getArrayRef());
        }
        return votes.getArrayCopy();
    }


    private int[][] kappaUpdateFirst(Instance instance, StreamingRandomPatches.StreamingRandomPatchesClassifier[] ensembleMembers) {
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

    private void kappaUpdateSecond(int[][] changeStatus, StreamingRandomPatches.StreamingRandomPatchesClassifier[] ensembleMembers,
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

    private HashMap<Integer, Integer> classifiersPredictions(Instance instance, StreamingRandomPatches.StreamingRandomPatchesClassifier[] ensembleMembers) {
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

//          TODO: This is internal to SRP, when we move to a generic approach, we may need to do that in here as well, but with a dedicated evaluator.
//          ensembleMembers[i].evaluator.addResult(example, vote.getArrayRef());
        }
        return classifiersExactPredictions;
    }



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

    // Debug
    private void estimateLearningAccuracy(Instance instance) {
        if(this.baseEnsemble != null && this.debugShowTTTAccuracyConfidenceOption.isSet()) {
            InstanceExample example = new InstanceExample(instance);
            double[] votes = this.baseEnsemble.getVotesForInstance(instance);
            this.evaluatorSupervisedDebug.addResult(example, votes);

            if(this.evaluatorEnsembleMembersDebug != null){
                for(int i = 0 ; i < this.baseEnsemble.getEnsembleMembers().length ; ++i) {
                    double[] voteEnsembleMember = this.baseEnsemble.getEnsembleMembers()[i].getVotesForInstance(instance);
                    this.evaluatorEnsembleMembersDebug[i].addResult(example, voteEnsembleMember);
                }
            }
        }
    }

    private void calculateDriftDetectionMetrics() {
        // TODO: implement metrics
        if(gtDriftLocations != null)
            for(int i = 0 ; i < this.gtDriftLocations.length ; ++i) {
                if(this.gtDriftLocations[i] == this.allInstancesSeen) {
                    this.outputDebugStream.println(this.allInstancesSeen + ", " + this.instancesSeen + ", "+ this.evaluatorSupervisedDebug.getPerformanceMeasurements()[1].getValue() +", CONCEPT DRIFT");
                }
            }
    }

    /*** KAPPA SUBNETWORKS ***/
//    public void knnSubnetworksKappa() {
//        selectSeedsKappa();
//        // Find the "complement" of the seed set, i.e. indexes of non-seed models.
//        ArrayList<Integer> complementSeeds = range(0, this.pairsOutputsKappa.length-1);
//        complementSeeds.removeAll(this.seedsKappa);
//        // Update the subnetworks vector assigning each seed with its own subnetwork.
//        for(int i : this.seedsKappa)
//            this.subnetworksKappa[i] = i;
//
//        // For all not in seed... decide to which seed they belong (max KAPPA)
//        for(int j : complementSeeds) {
//            // to init assume the first on seed to be the maximum
//            int maxKappaIdx = this.seedsKappa.get(0);
//
//            for(int i : this.seedsKappa) {
//                double kappa = getKappa(j, i);
//                double currentMaxKappa = getKappa(j, maxKappaIdx);
//
////                assert(!Double.isNaN(kappa) && !Double.isNaN(currentMaxKappa));
//
//                if(!Double.isNaN(kappa) && (Double.isNaN(currentMaxKappa) || kappa > currentMaxKappa))
//                    maxKappaIdx = i;
//            }
//            this.subnetworksKappa[j] = maxKappaIdx;
//        }
//    }

//    private void selectSeedsKappa() {
//        this.seedsKappa = new ArrayList<>();
//
//        int[] minKappa = idxMinKappa();
//        this.seedsKappa.add(minKappa[0]);
//        this.seedsKappa.add(minKappa[1]);
//
//        // Obtain the index that is "farther" from all currently selected.
//        while(this.seedsKappa.size() < this.numberOfSeedClassifiers.getValue()) {
//            int currentSelected = idxMinKappaFromSelected();
//            this.seedsKappa.add(currentSelected);
//        }
//    }

    protected double getKappa(int i, int j) {
        assert(i != j);
        if(this.pairsOutputsKappa == null)
            return -6;
        return i > j ? this.pairsOutputsKappa[j][i].k() : this.pairsOutputsKappa[i][j].k();
    }

//    private int[] idxMinKappa() {
//        int idx[] = new int[2];
//        idx[0] = 0;
//        idx[1] = 1;
//        for(int i = 0 ; i < this.pairsOutputsKappa.length ; ++i) {
//            for(int j = i+1 ; j < this.pairsOutputsKappa.length ; ++j) {
//                double kappa = getKappa(i,j);
//                double currentMinKappa = getKappa(idx[0], idx[1]);
//
//                if(!Double.isNaN(kappa) && (Double.isNaN(currentMinKappa) || kappa < currentMinKappa)) {
//                    idx[0] = i;
//                    idx[1] = j;
//                }
//            }
//        }
//        assert(idx[0] != idx[1]);
//        return idx;
//    }
//
//    private int idxMinKappaFromSelected() {
//        ArrayList<Integer> complementSeeds = range(0, this.pairsOutputsKappa.length-1);
//        complementSeeds.removeAll(this.seedsKappa);
//        double minSum = 100000.0;
//        int minIndex = -1;
//        for(int j : complementSeeds) {
//            double avg = averageForJColumnKappa(j);
//            double sum = sumForJColumnKappa(j, avg);
//
//            if(sum < minSum) {
//                minSum = sum;
//                minIndex = j;
//            }
//        }
////        System.out.println("MinIndexJ = " + minIndex);
//        return minIndex;
//    }

//    private double averageForJColumnKappa(int j) {
//        double sum = 0.0;
//        int count = 0;
//        for(int i : this.seedsKappa) {
//            double kappa = getKappa(i, j);
//            if(!Double.isNaN(kappa)) {
//                sum += kappa;
//                count++;
//            }
//        }
//        return sum/(double)count;
//    }

//    private double sumForJColumnKappa(int j, double avg) {
//        double sum = 0.0;
//        for(int i : this.seedsKappa) {
//            double kappa = getKappa(i, j);
//            if(!Double.isNaN(kappa)) {
//                sum += Math.abs(avg - kappa);
//            }
//        }
//        return sum;
//    }

//    public static ArrayList<Integer> range(int min, int max) {
//        ArrayList<Integer> list = new ArrayList<>();
//        for (int i = min; i <= max; i++) {
//            list.add(i);
//        }
//
//        return list;
//    }

//    protected class BaseModelStatistics {
//        public int index;
//        public int numInstancesSeen = 0; // total number of instances seen for training, either labeled or unlabeled
//        public int numInstancesLabeledSeen = 0;
//        public int numInstancesUnlabeledSeen = 0;
//        public int numCorrectInstancesAll = 0; // total number of correctly predicted instances, ignoring delay and unlabeled.
//        public int numCorrectInstancesLabeled = 0; // total number of correctly predicted instances, only those available to the learner during trainOnInstance()
//        public int numResets = 0; // total number of times this model has been reset
//        public int numPseudoLabelsUsedToTrainOthers = 0; // number of pseudo-labels used to train others
//        public int NumPseudoLabelsUsedToTrainOthersCorrect = 0; // number of pseudo-labels used to train others that were correct.
//        public int numPseudoLabelTrainedByOthers = 0; // number of pseudo-labeled instances used to train this
//        public int numPseudoLabelTrainedByOtherCorrect = 0; // ... correct
//
//        public BaseModelStatistics(int index) {
//            this.index = index;
//
//        }
//
//        public String getHeader() {
//            return "numInstancesSeen,numInstancesLabeledSeen,numInstancesUnlabeledSeen,numCorrectInstancesAll," +
//                    "numCorrectInstancesLabeled,numResets,numPseudoLabelsUsedToTrainOthers,NumPseudoLabelsUsedToTrainOthersCorrect," +
//                    "numPseudoLabelTrainedByOthers,numPseudoLabelTrainedByOtherCorrect";
//        }
//
//        /**
//         * Reset statistics that are not permanent, such as numInstancesSeen, numInstancesLabeled, ...
//         */
//        public void reset() {
//            numInstancesSeen = 0;
//            numInstancesLabeledSeen = 0;
//            numInstancesUnlabeledSeen = 0;
//            numCorrectInstancesAll = 0;
//            numCorrectInstancesLabeled = 0;
//        }
//    }

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
                //      System.out.println("column = (" + sum1 + "," + (sum1 / (double) instancesSeen) + ") row = (" + sum2 + "," + (sum2 / (double) instancesSeen) + ")");
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
