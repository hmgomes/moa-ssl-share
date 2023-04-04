package moa.tasks;

import moa.core.Measurement;

import static org.junit.Assert.*;

import org.junit.Test;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.SemiSupervisedLearner;

public class TestEvaluatedInterleavedTestThenTrainSSLDelayed {

    class MockLearner extends AbstractClassifier implements SemiSupervisedLearner {

        public int labeled;
        public int unlabeled;

        public MockLearner() {

        }

        @Override
        public int trainOnUnlabeledInstance(Instance instance) {
            this.unlabeled += 1;
            return -1; // no pseudo-label
        }

        @Override
        public void trainOnInstanceImpl(Instance inst) {
            this.labeled += 1;
        }

        @Override
        public void addInitialWarmupTrainingInstances() {
        }

        @Override
        public double[] getVotesForInstance(Instance inst) {
            return new double[0];
        }

        @Override
        public void resetLearningImpl() {
            return;
        }

        @Override
        public boolean isRandomizable() {
            return false;
        }

        @Override
        protected Measurement[] getModelMeasurementsImpl() {
            return new Measurement[0];
        }

        @Override
        public void getModelDescription(StringBuilder out, int indent) {
            throw new UnsupportedOperationException("Unimplemented method 'getModelDescription'");
        }

    }

    class MockLearnerWithExpectations extends MockLearner {

        public Boolean expect_labels[];
        public boolean is_warming_up;
        public int idx;
        public int warmup_length;

        public MockLearnerWithExpectations(Boolean expect_labels[], int warmup_length) {
            super();
            this.expect_labels = expect_labels;
            this.warmup_length = warmup_length;
            this.idx = -1;
        }

        @Override
        public int trainOnUnlabeledInstance(Instance instance) {
            this.idx += 1;
            // Called by unlabeled instances.
            int i = this.idx - this.warmup_length;
            assertTrue("All `trainOnUnlabeledInstance` calls should not have a class",
                    instance.isMissing(instance.classIndex()));
            assertFalse("Expected unlabeled instance at index " + i, this.expect_labels[i].booleanValue());
            return -1; // no pseudo-label
        }

        @Override
        public void trainOnInstanceImpl(Instance inst) {
            this.idx += 1;
            if (this.idx < this.warmup_length) {
                // Warmup instances must be labeled.
                assertFalse(inst.isMissing(inst.classIndex()));
            } else {
                // Labeled instances.
                int i = this.idx - this.warmup_length;
                assertFalse(inst.isMissing(inst.classIndex()));
                assertTrue("Expected labeled instance at index " + i, this.expect_labels[i].booleanValue());
            }
        }
    }

    private EvaluateInterleavedTestThenTrainSSLDelayed newSSL(
            SemiSupervisedLearner learner,
            int max_instances) {
        EvaluateInterleavedTestThenTrainSSLDelayed ssl = new EvaluateInterleavedTestThenTrainSSLDelayed();
        ssl.streamOption.setValueViaCLIString("generators.SEAGenerator");
        ssl.sslLearnerOption.setCurrentObject(learner);
        ssl.randomSeedOption.setValue(42);
        ssl.labelProbabilityOption.setValue(0.5);
        ssl.delayLengthOption.setValue(-1);
        ssl.initialWindowSizeOption.setValue(0);
        ssl.instanceLimitOption.setValue(max_instances);
        return ssl;
    }

    public void testSSLStream(int warmup, int delay) {
        /**
         * Ensure EvaluateInterleavedTestThenTrainSSLDelayed returns a
         * stream consistent with other implementations, of semi-supervised
         * learning, when the PRNG is seeded. This is to ensure that the
         * stream is deterministic and consistent.
         */
        Boolean[] expected_labels;

        /**
         * Expected outputs. Note that the number of instances varies based on
         * the delay length because instances remaining in the delay buffer
         * at the end of training are not returned. This is okay because
         * data streams are thought of as "infinite", therefore we don't need
         * to return the instances at the end because, there is not an end.
         */
        if (delay == -1)
            expected_labels = new Boolean[] { true, false, false, false, true, true, true, false, false, false };
        else if (delay == 0)
            expected_labels = new Boolean[] { true, false, true, false, true, false, true, true, true, true, false,
                    true, false, true, false };
        else if (delay == 1)
            expected_labels = new Boolean[] { true, false, false, true, true, false, true, true, true, true, false,
                    false, true, true, false };
        else if (delay == 5)
            expected_labels = new Boolean[] { true, false, false, false, true, true, true, true, true, true, false,
                    false, false };
        else if (delay == 100)
            expected_labels = new Boolean[] { true, false, false, false, true, true, true, false, false, false };
        else
            throw new RuntimeException("Unexpected delay value: " + delay);

        MockLearnerWithExpectations learner = new MockLearnerWithExpectations(expected_labels, warmup);
        EvaluateInterleavedTestThenTrainSSLDelayed ssl = newSSL(learner, 10 + warmup);
        ssl.initialWindowSizeOption.setValue(warmup);
        ssl.delayLengthOption.setValue(delay);
        ssl.prepareForUse();

        TaskMonitor monitor = new NullMonitor();
        ssl.doMainTask(monitor, null);

        assertEquals(expected_labels.length + warmup - 1, learner.idx);
    }

    @Test
    public void testSSLStreamNoDelayedLabels() {
        /** Test the case where there is no warmup and no delayed labels. */
        testSSLStream(0, -1);
        testSSLStream(100, -1);
    }

    @Test
    public void testSSLStreamDelay() {
        /**
         * Test the case where there is no warmup and labels are following
         * every unlabeled instance
         */
        testSSLStream(0, 0);
        testSSLStream(0, 1);
        testSSLStream(0, 5);
        testSSLStream(0, 100);
        testSSLStream(100, 0);
        testSSLStream(100, 1);
        testSSLStream(100, 5);
        testSSLStream(100, 100);
    }

    @Test
    public void testProbability() {
        MockLearner learner = new MockLearner();
        EvaluateInterleavedTestThenTrainSSLDelayed ssl = newSSL(learner, 10000);
        ssl.labelProbabilityOption.setValue(0.8);
        ssl.prepareForUse();

        TaskMonitor monitor = new NullMonitor();
        ssl.doMainTask(monitor, null);

        assertEquals(8038, learner.labeled);
        assertEquals(1962, learner.unlabeled);
    }
}
