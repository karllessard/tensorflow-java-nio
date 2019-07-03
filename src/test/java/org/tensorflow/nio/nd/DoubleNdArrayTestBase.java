/*
 Copyright 2019 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =======================================================================
 */
package org.tensorflow.nio.nd;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public abstract class DoubleNdArrayTestBase extends NdArrayTestBase<Double> {

    @Override
    protected abstract DoubleNdArray allocate(Shape shape);

    @Override
    protected Double valueOf(Long val) {
        return val.doubleValue();
    }

    @Test
    public void writeAndReadWithArrays() {
        double[] values = new double[] { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 };

        DoubleNdArray matrix = allocate(Shape.make(3, 4));
        matrix.write(values);
        assertEquals(Double.valueOf(0.0), matrix.get(0, 0));
        assertEquals(Double.valueOf(0.3), matrix.get(0, 3));
        assertEquals(Double.valueOf(0.4), matrix.get(1, 0));
        assertEquals(Double.valueOf(1.1), matrix.get(2, 3));

        matrix.set(100.5, 1, 0);
        matrix.read(values);
        assertEquals(0.0, values[0], 0.0);
        assertEquals(0.3, values[3], 0.0);
        assertEquals(100.5, values[4], 0.0);
        assertEquals(1.1, values[11], 0.0);
    }
}
