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

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Index;

public class DenseNdArray<T> extends AbstractDenseNdArray<T> {

  public static <T> NdArray<T> wrap(DataBuffer<T> buffer, Shape shape) {
    Validator.denseShape(shape);
    return new DenseNdArray<>(buffer, shape);
  }

  @Override
  public NdArray<T> at(long... indices) {
    return slice(indices, this::allocateSlice);
  }

  @Override
  public NdArray<T> slice(Index... indices) {
    return slice(indices, this::allocateSlice);
  }

  @Override
  protected DataBuffer<T> buffer() {
    return buffer;
  }

  private DenseNdArray(DataBuffer<T> buffer, Shape shape) {
    super(shape);
    this.buffer = buffer;
  }

  private DenseNdArray<T> allocateSlice(long position, Shape shape) {
    return new DenseNdArray<>(buffer.withPosition(position).slice(), shape);
  }

  private DataBuffer<T> buffer;
}