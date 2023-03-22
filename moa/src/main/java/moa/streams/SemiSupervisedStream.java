package moa.streams;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

import java.util.Objects;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

/**
 * This stream generator is a wrapper that takes any stream generator and
 * removes the label of randomly selected instances
 */
public class SemiSupervisedStream extends AbstractOptionHandler implements InstanceStream {

    /** The stream generator to be wrapped up */
    private InstanceStream stream;

    /** Type of stream generator to be used */
    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream generator to simulate semi-supervised setting", InstanceStream.class,
            "generators.SEAGenerator");

    /** Option: Probability of having a label */
    public FloatOption thresholdOption = new FloatOption("threshold", 't',
            "The ratio of unlabeled data",
            0.5);

    public IntOption initialWindowSizeOption = new IntOption("initialTrainingWindow", 'p',
            "Number of instances used for training in the beginning of the stream.",
            1000, 0, Integer.MAX_VALUE);

    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);

    /**
     * Random generator to obtain the probability of removing label from an
     * instance.
     * It is not seeded otherwise the probability will always be the same number
     */
    private RandomGenerator random;

    /** Probability that an instance will have a label */
    private double threshold;

    private static final long serialVersionUID = 1L;

    private long instancesProcessed = 0;

    @Override
    public String getPurposeString() {
        return "A wrapper that takes any stream generator and " +
                "removes the label of randomly selected instances to simulate semi-supervised setting";
    }

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        // sets up the threshold
        this.threshold = this.thresholdOption.getValue();

        this.instancesProcessed = 0;
        // sets up the stream generator
        this.stream = (InstanceStream) getPreparedClassOption(this.streamOption);
        if (this.stream instanceof AbstractOptionHandler) {
            ((AbstractOptionHandler) this.stream).prepareForUse(monitor, repository);
        }

        // Setup the random generator to use MT19937. This ensures consistency
        // across different platforms.
        this.random = new MersenneTwister(this.instanceRandomSeedOption.getValue());
    }

    @Override
    public InstancesHeader getHeader() {
        Objects.requireNonNull(this.stream, "The stream must not be null");
        return stream.getHeader();
    }

    @Override
    public long estimatedRemainingInstances() {
        Objects.requireNonNull(this.stream, "The stream must not be null");
        return stream.estimatedRemainingInstances();
    }

    @Override
    public boolean hasMoreInstances() {
        Objects.requireNonNull(this.stream, "The stream must not be null");
        return stream.hasMoreInstances();
    }

    @Override
    public Example<Instance> nextInstance() {
        Objects.requireNonNull(this.stream, "The stream must not be null");
        ++instancesProcessed;

        Example<Instance> inst = this.stream.nextInstance();

        // "instancesProcessed > this.initialWindowSizeOption.getValue()"
        // Check if this instance is not part of the first instances to guarantee
        // labeled data.
        // This is to ensure trainOnInitialWindow works properly.

        // if the probability is below the threshold, mask the label
        // i.e. the higher the probability, the more likely this instance is unlabeled
        // (and vice-versa)
        // ==> it corresponds to the ratio of unlabeled data
        if (instancesProcessed > this.initialWindowSizeOption.getValue()
                && this.random.nextDouble() >= this.threshold) {
            InstanceExample instEx = new InstanceExample(inst.getData().copy());
            // instEx.instance.setMissing(instEx.instance.classIndex());
            instEx.instance.setMasked(instEx.instance.classIndex());
            return instEx;
        }
        // else, just return the labeled instance
        return inst;
    }

    @Override
    public boolean isRestartable() {
        Objects.requireNonNull(this.stream, "The stream must not be null");
        return this.stream.isRestartable();
    }

    @Override
    public void restart() {
        Objects.requireNonNull(this.stream, "The stream must not be null");
        this.stream.restart();
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        // TODO
    }
}
