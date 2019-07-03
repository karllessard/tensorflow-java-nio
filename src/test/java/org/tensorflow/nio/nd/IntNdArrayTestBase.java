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

public abstract class IntNdArrayTestBase extends NdArrayTestBase<Integer> {

    @Override
    protected abstract IntNdArray allocate(Shape shape);

    @Override
    protected Integer valueOf(Long val) {
        return val.intValue();
    }

    @Test
    public void writeAndReadWithArrays() {
        int[] values = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        IntNdArray matrix = allocate(Shape.make(3, 4));
        matrix.write(values);
        assertEquals(Integer.valueOf(0), matrix.get(0, 0));
        assertEquals(Integer.valueOf(3), matrix.get(0, 3));
        assertEquals(Integer.valueOf(4), matrix.get(1, 0));
        assertEquals(Integer.valueOf(11), matrix.get(2, 3));

        matrix.set(100, 1, 0);
        matrix.read(values);
        assertEquals(0, values[0]);
        assertEquals(3, values[3]);
        assertEquals(100, values[4]);
        assertEquals(11, values[11]);
    }
}
