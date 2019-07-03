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
package org.tensorflow.nio.buffer;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import org.tensorflow.nio.buffer.single.ArrayDataBuffer;
import org.tensorflow.nio.buffer.single.ByteJdkDataBuffer;
import org.tensorflow.nio.buffer.single.DoubleJdkDataBuffer;
import org.tensorflow.nio.buffer.single.FloatJdkDataBuffer;
import org.tensorflow.nio.buffer.single.IntJdkDataBuffer;
import org.tensorflow.nio.buffer.single.LongJdkDataBuffer;
import org.tensorflow.nio.buffer.large.ByteLargeDataBuffer;
import org.tensorflow.nio.buffer.large.DoubleLargeDataBuffer;
import org.tensorflow.nio.buffer.large.FloatLargeDataBuffer;
import org.tensorflow.nio.buffer.large.IntLargeDataBuffer;
import org.tensorflow.nio.buffer.large.LargeDataBuffer;
import org.tensorflow.nio.buffer.large.LongLargeDataBuffer;

public final class DataBuffers {

  public static ByteDataBuffer ofBytes(long capacity) {
    if (capacity > ByteJdkDataBuffer.MAX_CAPACITY) {
      return ByteLargeDataBuffer.allocate(capacity);
    }
    return ByteJdkDataBuffer.allocate(capacity);
  }

  public static ByteDataBuffer wrap(byte[] array, boolean readOnly) {
    ByteBuffer buf = ByteBuffer.wrap(array);
    return ByteJdkDataBuffer.wrap(readOnly ? buf.asReadOnlyBuffer() : buf);
  }
  
  public static ByteDataBuffer wrap(ByteBuffer buf) {
    return ByteJdkDataBuffer.wrap(buf);
  }

  public static LongDataBuffer ofLongs(long capacity) {
    if (capacity > LongJdkDataBuffer.MAX_CAPACITY) {
      return LongLargeDataBuffer.allocate(capacity);
    }
    return LongJdkDataBuffer.allocate(capacity);
  }

  public static LongDataBuffer wrap(long[] array, boolean readOnly) {
    LongBuffer buf = LongBuffer.wrap(array);
    return LongJdkDataBuffer.wrap(readOnly ? buf.asReadOnlyBuffer() : buf);
  }
 
  public static LongDataBuffer wrap(LongBuffer buf) {
    return LongJdkDataBuffer.wrap(buf);
  }

  public static IntDataBuffer ofIntegers(long capacity) {
    if (capacity > IntJdkDataBuffer.MAX_CAPACITY) {
      return IntLargeDataBuffer.allocate(capacity);
    }
    return IntJdkDataBuffer.allocate(capacity);
  }

  public static IntDataBuffer wrap(int[] array, boolean readOnly) {
    IntBuffer buf = IntBuffer.wrap(array);
    return IntJdkDataBuffer.wrap(readOnly ? buf.asReadOnlyBuffer() : buf);
  }
 
  public static IntDataBuffer wrap(IntBuffer buf) {
    return IntJdkDataBuffer.wrap(buf);
  }

  public static DoubleDataBuffer ofDoubles(long capacity) {
    if (capacity > DoubleJdkDataBuffer.MAX_CAPACITY) {
      return DoubleLargeDataBuffer.allocate(capacity);
    }
    return DoubleJdkDataBuffer.allocate(capacity);
  }

  public static DoubleDataBuffer wrap(double[] array, boolean readOnly) {
    DoubleBuffer buf = DoubleBuffer.wrap(array);
    return DoubleJdkDataBuffer.wrap(readOnly ? buf.asReadOnlyBuffer() : buf);
  }

  public static DoubleDataBuffer wrap(DoubleBuffer buf) {
    return DoubleJdkDataBuffer.wrap(buf);
  }

  public static FloatDataBuffer ofFloats(long capacity) {
    if (capacity > FloatJdkDataBuffer.MAX_CAPACITY) {
      return FloatLargeDataBuffer.allocate(capacity);
    }
    return FloatJdkDataBuffer.allocate(capacity);
  }

  public static FloatDataBuffer wrap(float[] array, boolean readOnly) {
    FloatBuffer buf = FloatBuffer.wrap(array);
    return FloatJdkDataBuffer.wrap(readOnly ? buf.asReadOnlyBuffer() : buf);
  }

  public static FloatDataBuffer wrap(FloatBuffer buf) {
    return FloatJdkDataBuffer.wrap(buf);
  }

  public static <T> DataBuffer<T> of(Class<T> clazz, long capacity) {
    if (capacity > ArrayDataBuffer.MAX_CAPACITY) {
      return LargeDataBuffer.allocate(clazz, capacity);
    }
    return ArrayDataBuffer.allocate(clazz, capacity);
  }

  public static <T> DataBuffer<T> wrap(T[] array, boolean readOnly) {
    return ArrayDataBuffer.wrap(array, readOnly);
  }
}
