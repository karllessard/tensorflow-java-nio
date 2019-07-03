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

public abstract class FloatNdArrayTestBase extends NdArrayTestBase<Float> {

    @Override
    protected abstract FloatNdArray allocate(Shape shape);

    @Override
    protected Float valueOf(Long val) {
        return val.floatValue();
    }

    @Test
    public void writeAndReadWithArrays() {
        float[] values = new float[] { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f };

        FloatNdArray matrix = allocate(Shape.make(3, 4));
        matrix.write(values);
        assertEquals(Float.valueOf(0.0f), matrix.get(0, 0));
        assertEquals(Float.valueOf(0.3f), matrix.get(0, 3));
        assertEquals(Float.valueOf(0.4f), matrix.get(1, 0));
        assertEquals(Float.valueOf(1.1f), matrix.get(2, 3));

        matrix.set(100.5f, 1, 0);
        matrix.read(values);
        assertEquals(0.0f, values[0], 0.0);
        assertEquals(0.3f, values[3], 0.0);
        assertEquals(100.5f, values[4], 0.0);
        assertEquals(1.1f, values[11], 0.0);
    }
}
