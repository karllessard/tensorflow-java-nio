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
package org.tensorflow.nio.buffer.large;

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.ByteDataBufferTestBase;

public class ByteLargeDataBufferTest extends ByteDataBufferTestBase {

  @Override
  protected long maxCapacity() {
    return ByteLargeDataBuffer.MAX_CAPACITY;
  }

  @Override
  protected ByteDataBuffer allocate(long capacity) {
    return ByteLargeDataBuffer.allocate(capacity);
  }
}
