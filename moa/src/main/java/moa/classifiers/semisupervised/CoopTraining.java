package moa.classifiers.semisupervised;

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.SemiSupervisedLearner;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.Utils;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

public class CoopTraining extends AbstractClassifier implements SemiSupervisedLearner{

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classificador Base", AbstractClassifier.class,
            "moa.classifiers.trees.HoeffdingTree");

    public ClassOption subLearnerOption = new ClassOption("subLearner", 'z',
            "subClassificador", AbstractClassifier.class,
            "moa.classifiers.bayes.NaiveBayes");
    
    public FloatOption thresholdOption = new FloatOption("thresholdBase", 't',
            "Grau de confiança",
            2.0, 0.0, Double.MAX_VALUE);

	protected long initialWarmupTrainingCounter;
	protected long instancesPseudoLabeled;
	protected long instancesCorrectPseudoLabeled;

    private Classifier baseLearner;
    private Classifier subLearner;
    private double thresholdBase;
    int predictionBase;
    int predictionSub;
    double confidenceScoreBase;
	
    @Override
    public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        this.baseLearner = (Classifier) getPreparedClassOption(baseLearnerOption);
        this.subLearner = (Classifier) getPreparedClassOption(subLearnerOption);
        this.thresholdBase = thresholdOption.getValue();
        super.prepareForUseImpl(monitor, repository);
    }

    
	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		// TODO Auto-generated method stub
		return subLearner.getVotesForInstance(inst);
	}

	@Override
	public void resetLearningImpl() {
		// TODO Auto-generated method stub
		this.instancesCorrectPseudoLabeled = 0;
		this.instancesPseudoLabeled = 0;
	}

	@Override
	public void addInitialWarmupTrainingInstances() {
		++this.initialWarmupTrainingCounter;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// TODO Auto-generated method stub
//			if(!inst.classIsMasked() || !inst.classIsMissing()) {
//				baseLearner.trainOnInstance(inst);
//				subLearner.trainOnInstance(inst);
//			}
		baseLearner.trainOnInstance(inst);
		subLearner.trainOnInstance(inst);
	}

	@Override
	public int trainOnUnlabeledInstance(Instance instance) {

    	double[] votes = baseLearner.getVotesForInstance(instance);
		predictionBase = Utils.maxIndex(votes);
		confidenceScoreBase = 0.0;
		if(predictionBase >= 0 && votes != null && votes.length > 0)
			confidenceScoreBase = votes[predictionBase] / Utils.sum(votes);  //baseLearner.getConfidenceForPrediction(instance, predictionBase);

		predictionSub = Utils.maxIndex(subLearner.getVotesForInstance(instance));

		if(confidenceScoreBase > 0.0) {
			System.out.println();
		}

		if(predictionBase == predictionSub && confidenceScoreBase >= thresholdBase) {
			Instance instCopy = instance.copy();
			instCopy.setClassValue(predictionBase);
			subLearner.trainOnInstance(instCopy);

			if (!instance.classIsMissing() && ((int) instCopy.classValue()) == ((int) instance.classValue())) {
				++this.instancesCorrectPseudoLabeled;
			}
			++this.instancesPseudoLabeled;
		}
    	return 0;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return new Measurement[]{
				new Measurement("#pseudo-labeled", this.instancesPseudoLabeled),
				new Measurement("#correct pseudo-labeled", this.instancesCorrectPseudoLabeled),
				new Measurement("accuracy pseudo-labeled", this.instancesCorrectPseudoLabeled / (double) this.instancesPseudoLabeled * 100)
//                new Measurement("accuracy supervised learner", this.evaluatorSupervisedDebug.getPerformanceMeasurements()[1].getValue())
		};
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
		
	}

}
