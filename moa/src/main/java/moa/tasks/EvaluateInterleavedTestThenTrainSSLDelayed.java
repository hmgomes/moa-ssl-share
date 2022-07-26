package moa.tasks;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.SemiSupervisedLearner;
import moa.core.*;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.learners.Learner;
import moa.options.ClassOption;
import moa.streams.ExampleStream;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.LinkedList;
import java.util.Random;

/**
 * An evaluation task that relies on the mechanism of Interleaved Test Then Train,
 * applied on semi-supervised data streams
 */
public class EvaluateInterleavedTestThenTrainSSLDelayed extends SemiSupervisedMainTask {

    @Override
    public String getPurposeString() {
        return "Evaluates a classifier on a semi-supervised stream by testing only the labeled data, " +
                "then training with each example in sequence.";
    }

    private static final long serialVersionUID = 1L;

    public IntOption randomSeedOption = new IntOption(
            "instanceRandomSeed", 'r',
            "Seed for random generation of instances.", 1);

    public FlagOption onlyLabeledDataOption = new FlagOption("labeledDataOnly", 'a',
            "Learner only trained on labeled data");

    public ClassOption standarLearnerOption = new ClassOption("standardLearner", 'b',
            "A standard learner to train. This will be ignored if labeledDataOnly flag is not set.",
            MultiClassClassifier.class, "moa.classifiers.trees.HoeffdingTree");

    public ClassOption sslLearnerOption = new ClassOption("sslLearner", 'l',
            "A semi-supervised learner to train.", SemiSupervisedLearner.class,
            "moa.classifiers.semisupervised.ClusterAndLabelClassifier");

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to learn from.", ExampleStream.class,
            "SemiSupervisedStream");

    public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
            "Classification performance evaluation method.",
            LearningPerformanceEvaluator.class,
            "BasicClassificationPerformanceEvaluator");

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /** Option: Probability of instance being unlabeled */
    public FloatOption labelProbabilityOption = new FloatOption("labelProbability", 'j',
            "The ratio of unlabeled data",
            0.95);

    public IntOption delayLengthOption = new IntOption("delay", 'k',
            "Number of instances before test instance is used for training",
            1000, 1, Integer.MAX_VALUE);

    public IntOption initialWindowSizeOption = new IntOption("initialTrainingWindow", 'p',
            "Number of instances used for training in the beginning of the stream (-1 = no initialWindow).",
            1000, -1, Integer.MAX_VALUE);

//    public FlagOption trainOnInitialWindowOption = new FlagOption("trainOnInitialWindow", 'm',
//            "Whether to train or not using instances in the initial labeledInstancesBuffer.");

    public FlagOption debugPseudoLabelsOption = new FlagOption("debugPseudoLabels", 'w',
            "Learner only trained on labeled data");

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    public IntOption instanceLimitOption = new IntOption("instanceLimit", 'i',
            "Maximum number of instances to test/train on  (-1 = no limit).",
            100000000, -1, Integer.MAX_VALUE);

    public IntOption timeLimitOption = new IntOption("timeLimit", 't',
            "Maximum number of seconds to test/train for (-1 = no limit).", -1,
            -1, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency",
            'f',
            "How many instances between samples of the learning performance.",
            100000, 0, Integer.MAX_VALUE);

    public IntOption memCheckFrequencyOption = new IntOption(
            "memCheckFrequency", 'q',
            "How many instances between memory bound checks.", 100000, 0,
            Integer.MAX_VALUE);

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);

    public FileOption outputPredictionFileOption = new FileOption("outputPredictionFile", 'o',
            "File to append output predictions to.", null, "pred", true);

    private int numUnlabeledData = 0;

    private double labelProbability;

    /** Random Generator used  */
    public Random taskRandom;

    // Buffer of instances to use for training.
    protected LinkedList<Example> trainInstances;

    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
        String streamString = this.streamOption.getValueAsCLIString();
//        boolean isFullySupervised = this.onlyLabeledDataOption.isSet();

        this.trainInstances = new LinkedList<Example>();
        this.labelProbability = this.labelProbabilityOption.getValue();
        this.taskRandom = new Random(this.randomSeedOption.getValue());

        Learner learner;
        String learnerString;
        if(this.onlyLabeledDataOption.isSet()) {
            learner = (Learner) getPreparedClassOption(this.standarLearnerOption);
            learnerString = this.standarLearnerOption.getValueAsCLIString();
        } else {
            learner = (SemiSupervisedLearner) getPreparedClassOption(this.sslLearnerOption);
            learnerString = this.sslLearnerOption.getValueAsCLIString();
        }

        if (learner.isRandomizable()) {
            learner.setRandomSeed(this.randomSeedOption.getValue());
            learner.resetLearning();
        }
        ExampleStream stream = (ExampleStream) getPreparedClassOption(this.streamOption);

        LearningPerformanceEvaluator evaluator = (LearningPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        learner.setModelContext(stream.getHeader());


        int maxInstances = this.instanceLimitOption.getValue();
        long instancesProcessed = 0;
        int maxSeconds = this.timeLimitOption.getValue();
        int secondsElapsed = 0;
        monitor.setCurrentActivity("Evaluating learner...", -1.0);
        LearningCurve learningCurve = new LearningCurve(
                "learning evaluation instances");
        File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }
        //File for output predictions
        File outputPredictionFile = this.outputPredictionFileOption.getFile();
        PrintStream outputPredictionResultStream = null;
        if (outputPredictionFile != null) {
            try {
                if (outputPredictionFile.exists()) {
                    outputPredictionResultStream = new PrintStream(
                            new FileOutputStream(outputPredictionFile, true), true);
                } else {
                    outputPredictionResultStream = new PrintStream(
                            new FileOutputStream(outputPredictionFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open prediction result file: " + outputPredictionFile, ex);
            }
        }
        boolean firstDump = true;
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        long lastEvaluateStartTime = evaluateStartTime;
        double RAMHours = 0.0;

        while (stream.hasMoreInstances()
                && ((maxInstances < 0) || (instancesProcessed < maxInstances))
                && ((maxSeconds < 0) || (secondsElapsed < maxSeconds))) {

            // Obtain the next Example from the stream.
            // The instance is expected to be labeled.
            Example originalExample = stream.nextInstance();
            // Create a copy of the Example
            Example unlabeledExample = originalExample.copy();

            Instance originalInstanceData = (Instance) originalExample.getData();
            // Remove the label of the unlabeledExample indirectly through unlabeledInstanceData.
            Instance unlabeledInstanceData = (Instance) unlabeledExample.getData();

            // In case it is set, then the label is not removed. We want to pass the labelled data to the
            //     learner even in trainOnUnlabeled data to generate statistics such as number of correctly
            //     pseudo-labeled instances.
            if(! debugPseudoLabelsOption.isSet())
                unlabeledInstanceData.setMissing(unlabeledInstanceData.classIndex());



//            Example trainInst = stream.nextInstance();

            instancesProcessed++;

            // Train on the initial instances. These are not used for testing!
            if (instancesProcessed <= this.initialWindowSizeOption.getValue()) {
                if(learner instanceof SemiSupervisedLearner)
                    ((SemiSupervisedLearner) learner).addInitialWarmupTrainingInstances();
                learner.trainOnInstance(originalExample);
            } else {
                // Create yet another instance that will be used for training
                Example trainExample = originalExample.copy();
                Instance trainInstanceData = (Instance) trainExample.getData();


                // random = 0.01 and label probability = 0.05, do nothing.
                // random = 0.08 and label probability = 0.05, remove the label.
                double randomValue = this.taskRandom.nextDouble();
                if(this.labelProbability > randomValue) {
                    this.numUnlabeledData++;
                    trainInstanceData.setMissing(trainInstanceData.classIndex());
                }
                // Store instance for delayed training. This instance is labeled.
                this.trainInstances.addLast(trainExample);


                // Obtain the prediction for the testInst (i.e. no label)
                double[] prediction = learner.getVotesForInstance(unlabeledExample);


                // Train using the unlabeled instance.
                if(learner instanceof SemiSupervisedLearner)
                    ((SemiSupervisedLearner) learner).trainOnUnlabeledInstance(unlabeledInstanceData);

                // Output prediction
                if (outputPredictionFile != null) {
                    int trueClass = (int) originalInstanceData.classValue();
                    // Assuming that the class label is not missing for the originalInstanceData
                    outputPredictionResultStream.println(Utils.maxIndex(prediction) + "," + trueClass);
                }

                evaluator.addResult(originalExample, prediction);

                //  Use the oldest instance from the buffer of delayed for training.
                if(this.delayLengthOption.getValue() <= this.trainInstances.size()) {
                    // Only use it for training, if it is labeled.
                    // The unlabeled version was already used for training at the time the instance was generated.
                    Example oldestExample = this.trainInstances.removeFirst();
                    if( !((Instance) oldestExample.getData()).classIsMissing() )
                        learner.trainOnInstance(oldestExample);
                }

                if (instancesProcessed % this.sampleFrequencyOption.getValue() == 0 || !stream.hasMoreInstances()) {
                    long evaluateTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                    double time = TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime);
                    double timeIncrement = TimingUtils.nanoTimeToSeconds(evaluateTime - lastEvaluateStartTime);
                    double RAMHoursIncrement = learner.measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
                    RAMHoursIncrement *= (timeIncrement / 3600.0); //Hours
                    RAMHours += RAMHoursIncrement;
                    lastEvaluateStartTime = evaluateTime;
                    learningCurve.insertEntry(new LearningEvaluation(
                            new Measurement[]{
                                    new Measurement(
                                            "learning evaluation instances",
                                            instancesProcessed),
                                    new Measurement(
                                            "evaluation time ("
                                                    + (preciseCPUTiming ? "cpu "
                                                    : "") + "seconds)",
                                            time),
                                    new Measurement(
                                            "model cost (RAM-Hours)",
                                            RAMHours),
                                    new Measurement(
                                            "Unlabeled instances",
                                            this.numUnlabeledData
                                    )
                            },
                            evaluator, learner));
                    if (immediateResultStream != null) {
                        if (firstDump) {
                            immediateResultStream.print("Learner,stream,randomSeed,");
                            immediateResultStream.println(learningCurve.headerToString());
                            firstDump = false;
                        }
                        immediateResultStream.print(learnerString + "," + streamString + ","
                                + this.randomSeedOption.getValueAsCLIString() + ",");
                        immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                        immediateResultStream.flush();
                    }
                }
                if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                    if (monitor.taskShouldAbort()) {
                        return null;
                    }
                    long estimatedRemainingInstances = stream.estimatedRemainingInstances();
                    if (maxInstances > 0) {
                        long maxRemaining = maxInstances - instancesProcessed;
                        if ((estimatedRemainingInstances < 0)
                                || (maxRemaining < estimatedRemainingInstances)) {
                            estimatedRemainingInstances = maxRemaining;
                        }
                    }
                    monitor.setCurrentActivityFractionComplete(estimatedRemainingInstances < 0 ? -1.0
                            : (double) instancesProcessed / (double) (instancesProcessed + estimatedRemainingInstances));
                    if (monitor.resultPreviewRequested()) {
                        monitor.setLatestResultPreview(learningCurve.copy());
                    }
                    secondsElapsed = (int) TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()
                            - evaluateStartTime);
                }
            }
        }
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
        if (outputPredictionResultStream != null) {
            outputPredictionResultStream.close();
        }
        return learningCurve;
    }

    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }
}
