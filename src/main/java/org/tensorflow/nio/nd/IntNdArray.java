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

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.nd.index.Index;

public interface IntNdArray extends NdArray<Integer> {
  
  @Override
  IntNdArray at(long... indices);
  
  @Override
  IntNdArray slice(Index... indices);

  @Override
  Iterable<IntNdArray> childElements();

  @Override
  IntNdArray set(Integer value, long... indices);

  @Override
  IntNdArray copyTo(NdArray<Integer> dst);

  @Override
  IntNdArray copyFrom(NdArray<Integer> src);

  @Override
  IntNdArray read(DataBuffer<Integer> dst);

  @Override
  IntNdArray write(DataBuffer<Integer> src);

  default void read(int[] dst) { read(DataBuffers.wrap(dst, false)); }
  
  default void write(int[] src) { write(DataBuffers.wrap(src, false)); }
}
