package moa.classifiers.semisupervised;

import com.yahoo.labs.samoa.instances.Instance;

import java.util.Random;

public interface SLEADEBaseClassifier {
    public void trainOnInstance(Instance instance, double weight, long instancesSeen, Random random, boolean pseudoLabeledInstance);
}
