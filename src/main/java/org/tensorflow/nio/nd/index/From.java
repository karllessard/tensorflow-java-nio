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

import org.tensorflow.nio.nd.dimension.Dimension;

class From implements Index {
  
  public From(long start) {
    this.start = start;
  }

  @Override
  public long numElements(Dimension dim) {
    return dim.numElements() - start;
  }

  @Override
  public long mapPosition(long elementIndex, Dimension dim) {
    return start + elementIndex;
  }
  
  private long start;
}
