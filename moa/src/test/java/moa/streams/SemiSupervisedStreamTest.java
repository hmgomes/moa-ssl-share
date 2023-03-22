package moa.streams;

import moa.core.Example;
import static org.junit.Assert.*;

import org.junit.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import com.yahoo.labs.samoa.instances.Instance;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

public class SemiSupervisedStreamTest {

    @Test
    public void testPRNGConsistency() {
        /**
         * Ensure the PRNG is seeded correctly, so that the stream
         * is deterministic and consistent with other languages.
         */
        double expected[] = { 0.37454, 0.95071, 0.73199, 0.59865, 0.15601, 0.15599, 0.05808, 0.86617, 0.60111,
                0.70807 };
        RandomGenerator prng = new MersenneTwister(42);

        for (int i = 0; i < expected.length; i++) {
            double actual = prng.nextDouble();
            assertEquals(expected[i], actual, 0.00001);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = { 0, 100 })
    public void testStreamConsistency(int warmup) {
        /**
         * Ensure SemiSupervisedStream returns a consistent stream of
         * labeled vs unlabeled instances, when the PRNG is seeded. This
         * is to ensure that the stream is deterministic and consistent
         * with other languages.
         */
        boolean expected_output[] = { true, false, false, false, true, true, true, false, false, false };
        SemiSupervisedStream stream = new SemiSupervisedStream();
        stream.streamOption.setValueViaCLIString("generators.SEAGenerator");
        // PRNG should not be affected by the warmup period.
        stream.initialWindowSizeOption.setValue(warmup);
        stream.thresholdOption.setValue(0.5);
        stream.instanceRandomSeedOption.setValue(42);
        stream.prepareForUse();

        // consume stream
        for (int i = 0; i < expected_output.length + warmup; i++) {
            Example<Instance> inst = stream.nextInstance();
            boolean has_label = !inst.getData().classIsMasked();
            // ignore warmup
            if (i < warmup) {
                assertTrue("Instances in the warmup period must have labels", has_label);
                continue;
            }
            assertEquals(expected_output[i - warmup], has_label);
        }
    }

    @Test
    public void testProbability() {
        /**
         * Ensure the frequency of labeled vs unlabeled instances
         * is approximately correct.
         */

        SemiSupervisedStream stream = new SemiSupervisedStream();
        stream.streamOption.setValueViaCLIString("generators.SEAGenerator");
        stream.initialWindowSizeOption.setValue(0);
        stream.thresholdOption.setValue(0.8);
        stream.instanceRandomSeedOption.setValue(42);
        stream.prepareForUse();

        // consume stream
        int num_labeled = 0;
        int num_unlabeled = 0;
        int n = 10000;
        for (int i = 0; i < n; i++) {
            Example<Instance> inst = stream.nextInstance();

            // check if the instance has a label
            boolean has_label = !inst.getData().classIsMasked();
            if (has_label) {
                num_labeled++;
            } else {
                num_unlabeled++;
            }
        }

        // Because the seed is fixed, we can check the exact number of
        // labeled and unlabeled instances.
        assertEquals("Roughly 80% of the instances should be labeled",
                8038, num_labeled);
        assertEquals("Roughly 20% of the instances should be unlabeled",
                1962, num_unlabeled);
    }
}
