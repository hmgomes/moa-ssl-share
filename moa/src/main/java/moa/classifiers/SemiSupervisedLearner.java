/*
 *    SemiSupervisedLearner.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers;

import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Example;
import moa.learners.Learner;

/**
 * Learner interface for incremental semi supervised models. It is used only in the GUI Regression Tab. 
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public interface SemiSupervisedLearner extends Learner<Example<Instance>> {
    // Returns the pseudo-label used. If no pseudo-label was used, then return -1.
    int trainOnUnlabeledInstance(Instance instance);

    void addInitialWarmupTrainingInstances();
}
