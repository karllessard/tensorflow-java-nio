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
package org.tensorflow.nio.nd.dense;

import java.util.function.BiFunction;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.nd.impl.AbstractNdArray;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.IllegalRankException;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Index;

public abstract class AbstractDenseNdArray<T> extends AbstractNdArray<T> {

  @Override
  public T get(long... indices) {
    return buffer().get(position(indices, true));
  }

  @Override
  public void set(T value, long... indices) {
    buffer().put(position(indices, true), value);
  }

  @Override
  public void copyTo(NdArray<T> dst) {
    // TODO Optimize when array is continuous in memory
    super.copyTo(dst);
  }

  @Override
  public void copyFrom(NdArray<T> src) {
    // TODO Optimize when array is continuous in memory
    super.copyFrom(src);
  }

  @Override
  public void read(DataBuffer<T> dst) {
    if (isBulkCopyAvailable()) {
      BulkDataTransfer.create(this).execute(b -> dst.put(b));
    } else {
      super.read(dst);
    }
  }

  @Override
  public void write(DataBuffer<T> src) {
    if (isBulkCopyAvailable()) {
      BulkDataTransfer.create(this).execute(b -> b.put(src));
    } else {
      super.write(src);
    }
  }

  protected AbstractDenseNdArray(Shape shape) {
    super(shape);
  }

  protected abstract DataBuffer<T> buffer();

  protected <U extends NdArray<T>> U slice(long[] indices, BiFunction<Long, Shape, U> sliceAllocator) {
    Shape sliceShape = shape().subshape(indices.length);
    long slicePosition = position(indices, false);
    return sliceAllocator.apply(slicePosition, sliceShape);
  }
  
  protected <U extends NdArray<T>> U slice(Index[] indices, BiFunction<Long, Shape, U> sliceAllocator) {
    Shape sliceShape = shape().mapTo(indices);
    long slicePosition = 0L;
    int i = 0;
    while (i < sliceShape.numDimensions() && sliceShape.dimension(i).numElements() == 0) {
      slicePosition += sliceShape.dimension(i++).position();
    }
    if (i > 0) {
      sliceShape = sliceShape.subshape(i);
    }
    return sliceAllocator.apply(slicePosition, sliceShape);
  }
  
  private long position(long[] indices, boolean scalar) {
    if (indices.length > shape().numDimensions()) {
      throw new IndexOutOfBoundsException();
    }
    long position = 0L;
    int i = 0;
    for (; i < indices.length; ++i) {
      position += shape().dimension(i).positionOf(indices[i]);
    }
    while (i < shape().numDimensions() && shape().dimension(i).numElements() == 0) {
      position += shape().dimension(i++).position();
    }
    if (scalar && i < shape().numDimensions()) {
      throw new IllegalRankException("Not a scalar value");
    }
    return position;
  }

  /**
   * Check if we copy this array data in bulk. Bulk copy is only possible for array of 1-dimension or more and that
   * the last dimension is not segmented (therefore linear in memory).
   *
   * @return true if bulk copy is possible
   */
  private boolean isBulkCopyAvailable() {
    return shape().numDimensions() > 0 && !shape().dimension(shape().numDimensions() - 1).isSegmented();
  }
}
