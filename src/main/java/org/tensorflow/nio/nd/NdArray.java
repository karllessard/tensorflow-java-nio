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
import org.tensorflow.nio.nd.index.Index;
import org.tensorflow.nio.nd.iterator.ValueIterable;
import org.tensorflow.nio.nd.iterator.ValueIterator;

/**
 * A data structure of N-dimensions.
 *
 * <p>The `NdArray` interface creates an abstraction between the physical storage of a data record,
 * which can be linear or segmented, and its logical representation. In general, they achieve
 * better performances than standard multi-dimensional arrays in Java by mapping directly the
 * data in memory.
 *
 * <p>Like {@link DataBuffer}, {@code NdArray} instances support 64-bits indexation so they can be
 * used to map very large data records. They also support special indices that to traverse their
 * values in any direction or to select only a subset of them.
 *
 * <p>Example of usage:
 * <pre>{@code
 *    // Creates a 3x2x2 matrix (of rank 3)
 *    FloatNdArray matrix3d = NdArrays.ofFloats(Shape.make(3, 2, 2));
 *
 *    // Access the second 2x2 matrix (of rank 2)
 *    FloatNdArray matrix = matrix3d.at(1);
 *
 *    // Initialize second matrix data with an array of floats
 *    matrix.write(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
 *
 *    // Access directly the float value at (1, 1, 0) from the 3D-matrix
 *    assertEquals(3.0f, matrix3d.get(1, 1, 0));
 * }</pre>
 *
 * @param <T> the type of values to be mapped
 */
public interface NdArray<T> {

  /**
   * @return the shape of this N-dimensional array
   */
  Shape shape();

  /**
   * Computes and returns the total size of this N-dimensional array, in number of values.
   *
   * <p>For example, given a 3x3x2 matrix, the return value will be 18.
   * @return total size of this nd array
   */
  long size();

  /**
   * Returns an iteration of the elements of the first dimension of this N-dimensional array.
   *
   * <p>For example, given a 3x3x2 matrix, the return value can be used to iterates the three 3x2
   * matrices,which are {@code array.at(0)}, {@code array.at(1)} and {@code array.at(2)}.
   *
   * @return an iteration of N-dimensional arrays
   * @throws IllegalRankException if this array is a scalar (rank 0)
   */
  Iterable<? extends NdArray<T>> childElements();

  /**
   * Returns an iteration of all values of this N-dimension array.
   *
   * <p>Values necessarily resides in the last dimension of the array, so the number of values that
   * could be visited by this iteration is equal to {@link #size()}.
   *
   * <p>The return iterable can be used to access a {@link  ValueIterator} directly that allows
   * visiting the values for assignation. For example:
   * <pre>{@code
   *    // Access values for output
   *    for (Float value: arrayOfFloat.values()) {
   *      System.out.println(value);
   *    }
   *    // Access values for input
   *    float val = 0.0f;
   *    for (ValueIterator<Float> iter = arrayOfFloat.values().iterator(); iter.hasNext();) {
   *      iter.next(val++);
   *    }
   * }</pre>
   * @return an iteration of values of type {@code T}
   * @throws IllegalRankException if this array is a scalar (rank 0)
   */
  ValueIterable<T> values();

  /**
   * Returns the N-dimensional element of this array at the given coordinates.
   *
   * <p>Elements of all dimensions are accessible through this method, meaning that if the number of
   * indices provided is equal to the number of dimensions, then a rank-0 (scalar) array is
   * returned, which value can then be obtained with `array.get()`.
   *
   * <p>Any changes applied to the returned elements affect the data of this array as well, as there
   * is no copy involved.
   *
   * @param indices coordinates of the element to access, none will return this array
   * @return the element at this index
   */
  NdArray<T> at(long... indices);

  /**
   * Creates a multi-dimensional view (or slice) of this array by mapping one or more dimensions
   * to the given index selectors.
   *
   * <p>Slices allow to traverse an N-dimensional array in any of its axis and to filter only
   * elements of interest. For example, for a given matrix on the {@code [x, y]} axes, it is
   * possible to iterate through all rows on the {@code x} axis or the last
   * three columns on the {@code y}.
   *
   * <p>Any changes applied to the returned slice affect the data of this array as well, as there
   * is no copy involved.
   *
   * <p>Example of usage:
   * <pre>{@code
   *    import static org.tensorflow.nio.nd.index.Indices.*;
   *
   *    NdArray<Float> matrix3d = NdArrays.ofFloats(Shape.make(3, 2, 4));  // with [x, y, z] axes
   *
   *    // Iterates values over the 3rd elements on the z axis, (i.e. [x, x, 2])
   *    for (Float values = matrix3d.slice(all(), all(), at(2)).values()) {
   *      ...
   *    }
   *
   *    // Creates a slice that contains only the last element of the y axis and elements with an
   *    // odd `z` coordinate.
   *    NdArray<Float> slice = matrix3d.slice(all(), at(1), odd());
   *    assertEquals(Shape.make(3, 2), slice.shape());  // x=3, y=0 (scalar), z=2 (odd coordinates)
   *
   *    // Iterates backward the elements on the x axis
   *    for (NdArray<Float> matrix = matrix3d.slice(flip())) {
   *      assertEquals(Shape.make(2, 4), matrix);  // y=2, z=4
   *    }
   * }</pre>
   *
   * @param indices index selectors per dimensions, starting from dimension 0 of this array.
   * @return the element resulting of the index selection
   */
  NdArray<T> slice(Index... indices);

  T get(long... indices);

  void set(T value, long... indices);

  void copyTo(NdArray<T> dst);

  void copyFrom(NdArray<T> src);

  void read(DataBuffer<T> dst);

  void write(DataBuffer<T> src);
}
