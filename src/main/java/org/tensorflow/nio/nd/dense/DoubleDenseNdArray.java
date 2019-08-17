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

import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.nd.DoubleNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Index;

public class DoubleDenseNdArray extends AbstractDenseNdArray<Double> implements DoubleNdArray {

  public static DoubleNdArray wrap(DoubleDataBuffer buffer, Shape shape) {
    Validator.denseShape(shape);
    return new DoubleDenseNdArray(buffer, shape);
  }

  @Override
  @SuppressWarnings({"unchecked", "raw"})
  public Iterable<DoubleNdArray> childElements() {
    return (Iterable)super.childElements();
  }

  @Override
  public DoubleNdArray at(long... indices) {
    return slice(indices, this::allocateSlice);
  }

  @Override
  public DoubleNdArray slice(Index... indices) {
    return slice(indices, this::allocateSlice);
  }

  @Override
  protected DoubleDataBuffer buffer() {
    return buffer;
  }

  private DoubleDenseNdArray(DoubleDataBuffer buffer, Shape shape) {
    super(shape);
    this.buffer = buffer;
  }

  private DoubleDenseNdArray allocateSlice(long position, Shape shape) {
    return new DoubleDenseNdArray(buffer.withPosition(position).slice(), shape);
  }

  private DoubleDataBuffer buffer;
}