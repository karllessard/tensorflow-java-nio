package org.tensorflow.nio;

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.nd.ByteNdArray;
import org.tensorflow.nio.nd.DoubleNdArray;
import org.tensorflow.nio.nd.FloatNdArray;
import org.tensorflow.nio.nd.IntNdArray;
import org.tensorflow.nio.nd.LongNdArray;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.NdArrays;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Index;
import org.tensorflow.nio.nd.index.Indices;

final class StaticImports {

  public static <T> DataBuffer<T> bufferOf(Class<T> clazz, long capacity) {
    return DataBuffers.of(clazz, capacity);
  }

  public static <T> DataBuffer<T> bufferOf(T[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }
  
  public static ByteDataBuffer bufferOfBytes(long capacity) {
    return DataBuffers.ofBytes(capacity);
  }

  public static ByteDataBuffer bufferOf(byte[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  public static IntDataBuffer bufferOfInts(long capacity) {
    return DataBuffers.ofIntegers(capacity);
  }

  public static IntDataBuffer bufferOf(int[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }
  
  public static LongDataBuffer bufferOfLongs(long capacity) {
    return DataBuffers.ofLongs(capacity);
  }

  public static LongDataBuffer bufferOf(long[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }
  
  public static FloatDataBuffer bufferOfFloats(long capacity) {
    return DataBuffers.ofFloats(capacity);
  }

  public static FloatDataBuffer bufferOf(float[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }
  
  public static DoubleDataBuffer bufferOfDoubles(long capacity) {
    return DataBuffers.ofDoubles(capacity);
  }

  public static DoubleDataBuffer bufferOf(double[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  public static <T> NdArray<T> ndArrayOf(Class<T> clazz, Shape shape) {
    return NdArrays.of(clazz, shape);
  }

  public static <T> NdArray<T> ndArrayOf(T[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  public static <T> NdArray<T> ndArrayOf(DataBuffer<T> buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  } 

  public static ByteNdArray ndArrayOfBytes(Shape shape) {
    return NdArrays.ofBytes(shape);
  }

  public static ByteNdArray ndArrayOf(byte[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  public static ByteNdArray ndArrayOf(ByteDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  public static IntNdArray ndArrayOfInts(Shape shape) {
    return NdArrays.ofIntegers(shape);
  }

  public static IntNdArray ndArrayOf(int[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  public static IntNdArray ndArrayOf(IntDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  public static LongNdArray ndArrayOfLongs(Shape shape) {
    return NdArrays.ofLongs(shape);
  }

  public static LongNdArray ndArrayOf(long[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  public static LongNdArray ndArrayOf(LongDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  public static FloatNdArray ndArrayOfFloats(Shape shape) {
    return NdArrays.ofFloats(shape);
  }

  public static FloatNdArray ndArrayOf(float[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  public static FloatNdArray ndArrayOf(FloatDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  public static DoubleNdArray ndArrayOfDoubles(Shape shape) {
    return NdArrays.ofDoubles(shape);
  }

  public static DoubleNdArray ndArrayOf(double[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  public static DoubleNdArray ndArrayOf(DoubleDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  public static Index at(long index) {
    return Indices.at(index);
  }

  public static Index at(NdArray<? extends Number> index) {
    return Indices.at(index);
  }
  
  public static Index all() {
    return Indices.all();
  }
  
  public static Index seq(long... indices) {
    return Indices.seq(indices);
  }
  
  public static Index elem(NdArray<? extends Number> indices) {
    return Indices.elem(indices);
  }
  
  public static Index even() {
    return Indices.even();
  }

  public static Index odd() {
    return Indices.odd();
  }
  
  public static Index step(long stepLength) {
    return Indices.step(stepLength);
  }
  
  public static Index from(long start) {
    return Indices.from(start);
  }

  public static Index to(long end) {
    return Indices.to(end);
  }
  
  public static Index range(long start, long end) {
    return Indices.range(start, end);
  }
}
