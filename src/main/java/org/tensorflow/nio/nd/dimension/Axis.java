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

final class Axis extends AbstractDimension {
  
  Axis(long numElements, long positionStep) {
    this.numElements = numElements;
    this.positionStep = positionStep;
  }
  
  @Override
  public long numElements() {
    return numElements;
  }
  
  @Override
  public long positionOf(long elementIndex) {
    if (elementIndex >= numElements) {
      throw new IndexOutOfBoundsException();
    }
    return positionStep * elementIndex;
  }

  @Override
  public boolean isSegmented() {
    return false;  // all axis are continuous
  }

  @Override
  long positionStep() {
    return positionStep;
  }
  
  @Override
  public String toString() {
    return String.valueOf(numElements);
  }

  private final long numElements;
  private final long positionStep;
}
