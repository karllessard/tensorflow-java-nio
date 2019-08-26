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
package org.tensorflow.nio.nd.index;

import org.tensorflow.nio.nd.impl.dimension.Dimension;
import org.tensorflow.nio.nd.impl.dimension.Dimensions;

/**
 * An index that returns only elements found at an even position in the original dimension.
 *
 * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and n is even,
 * {@code even()} returns x<sub>0</sub>, x<sub>2</sub>, ..., x<sub>n-2</sub>
 */
class Even implements Index {

  static final Even INSTANCE = new Even();

  @Override
  public long numElements(Dimension dim) {
    return (dim.numElements() >> 1) + 1;
  }

  @Override
  public long mapCoordinate(long coordinate, Dimension dim) {
    return coordinate << 1;
  }

  private Even() {
  }
}
