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

import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.NdArrays;
import org.tensorflow.nio.nd.Shape;

/**
 * Helper class for creating {@link Index} instances.
 */
public final class Indices {

  /**
   * An index that selects a specific element on a given dimension.
   *
   * <p>When this index is applied to a given dimension, the dimension is resolved as a
   * single element and therefore is excluded from the computation of the rank.
   *
   * <p>For example, given a 3D matrix on the axis [x, y, z], if
   * {@code matrix.slice(all(), at(0), at(0)}, then the rank of the returned slice is 1 and its
   * number of elements is {@code x.numElements()}
   *
   * @param coordinate coordinate of the element referenced by this index
   * @return an index
   */
  public static Index at(long coordinate) {
    return new At(coordinate);
  }

  /**
   * An index that selects a specific element on a given dimension, as provided by a scalar
   * coordinate.
   *
   * <p>This is the equivalent of calling {@code at(scalar.get().longValue())}
   *
   * @param scalar scalar providing the coordinate of the element referenced by this index
   * @return an index
   * @see #at(long)
   */
  public static Index at(NdArray<? extends Number> scalar) {
    if (scalar.shape().numDimensions() > 0) {
      throw new IllegalArgumentException("Only scalars are accepted as a value index");
    }
    return new At(scalar.get().longValue());
  }

  /**
   * An index that returns all elements of a dimension in the original order.
   *
   * <p>Applying this index to a given dimension will return the original dimension
   * directly.
   *
   * <p>For example, given a vector with {@code n} elements, {@code all()} returns
   * x<sub>0</sub>, x<sub>1</sub>, ..., x<sub>n-1</sub>
   *
   * @return an index
   */
  public static Index all() {
    return All.INSTANCE;
  }

  /**
   * An index that returns only specific elements on a given dimension.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and {@code n >
   * 10}, {@code seq(8, 0, 3)} returns x<sub>8</sub>, x<sub>0</sub>, x<sub>3</sub>
   *
   * @param coordinates coordinates of the element referenced by this index, in the sequence order
   * @return an index
   */
  public static Index seq(long... coordinates) {
    if (coordinates == null) {
      throw new IllegalArgumentException();
    }
    return new Sequence(NdArrays.wrap(coordinates, Shape.create(coordinates.length)));
  }

  /**
   * An index that returns only specific elements on a given dimension, as provided by a vector of
   * coordinates.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and {@code n >
   * 10}, {@code seq(8, 0, 3)} returns x<sub>8</sub>, x<sub>0</sub>, x<sub>3</sub>
   *
   * @param vector vector of coordinates of the element referenced by this index, in the sequence
   *               order
   * @return an index
   */
  public static Index seq(NdArray<? extends Number> vector) {
    if (vector.shape().numDimensions() != 1) {
      throw new IllegalArgumentException("Only vectors are accepted as an element index");
    }
    return new Sequence(vector);
  }

  /**
   * An index that returns only elements found at an even position in the original dimension.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and n is even,
   * {@code even()} returns x<sub>0</sub>, x<sub>2</sub>, ..., x<sub>n-2</sub>
   *
   * @return an index
   */
  public static Index even() {
    return Even.INSTANCE;
  }

  /**
   * An index that returns only elements found at an odd position in the original dimension.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and n is even,
   * {@code odd()} returns x<sub>1</sub>, x<sub>3</sub>, ..., x<sub>n-1</sub>
   *
   * @return an index
   */
  public static Index odd() {
    return Odd.INSTANCE;
  }

  /**
   * An index that returns only elements on a given dimension starting at a specific coordinate.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and
   * {@code n > k}, {@code from(k)} returns x<sub>k</sub>, x<sub>k+1</sub>, ..., x<sub>n-1</sub>
   *
   * @param start coordinate of the first element referenced by this index
   * @return an index
   * @throws IllegalArgumentException if start is negative
   */
  public static Index from(long start) {
    if (start < 0) {
      throw new IllegalArgumentException("Start coordinate cannot be negative");
    }
    return new From(start);
  }

  /**
   * An index that returns only elements on a given dimension up to a specific coordinate.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and
   * {@code n > k}, {@code to(k)} returns x<sub>0</sub>, x<sub>1</sub>, ..., x<sub>k</sub>
   *
   * @param end coordinate of the last element referenced by this index
   * @return an index
   * @throws IllegalArgumentException if end is negative
   */
  public static Index to(long end) {
    if (end < 0) {
      throw new IllegalArgumentException("End coordinate cannot be negative");
    }
    return new To(end);
  }

  /**
   * An index that returns only elements on a given dimension between two coordinates.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and
   * {@code n > k > j}, {@code range(j, k)} returns x<sub>j</sub>, x<sub>j+1</sub>, ...,
   * x<sub>k</sub>
   *
   * @param start coordinate of the first element referenced by this index
   * @param end coordinate of the last element referenced by this index
   * @return an index
   * @throws IllegalArgumentException if start or end is negative, or if start is greated than end
   */
  public static Index range(long start, long end) {
    if (start < 0) {
      throw new IllegalArgumentException("Start coordinate cannot be negative");
    }
    if (end < 0) {
      throw new IllegalArgumentException("End coordinate cannot be negative");
    }
    if (start > end) {
      throw new IllegalArgumentException("Start coordinate cannot be greater than end coordinate");
    }
    return new Range(start, end);
  }

  /**
   * An index that reverse the order of the elements on a given dimension.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis,
   * this index returns x<sub>n-1</sub>, x<sub>n-2</sub>, ..., x<sub>0</sub>
   *
   * @return an index
   */
  public static Index flip() {
    return Flip.INSTANCE;
  }
}
