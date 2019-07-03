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
package org.tensorflow.nio.nd.dimension;

import org.tensorflow.nio.nd.index.Index;

final class IndexedDimension extends AbstractDimension {
  
  IndexedDimension(Index index, AbstractDimension dim) {
    this.index = index;
    this.dimension = dim;
  }
  
  @Override
  public long numElements() {
    return index.numElements(dimension);
  }
  
  @Override
  public long positionOf(long elementIndex) {
    if (elementIndex >= numElements()) {
      throw new IndexOutOfBoundsException();
    }
    return dimension.positionOf(index.mapPosition(elementIndex, dimension));
  }
  
  @Override
  public boolean isSegmented() {
    // TODO for now we consider all indexed dimensions as segmented but might depend on the actual index
    return true;
  }

  @Override
  long positionStep() {
    return dimension.positionStep();
  }

  @Override
  public String toString() {
    return String.valueOf(numElements());
  }

  private final Index index;
  private final AbstractDimension dimension;
}
