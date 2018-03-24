import DType from '../dtype/interface';
import Graph from '../graph/interface';
import Operation from '../operation/interface';
import TensorShape from '../tensorShape/interface';
import SparseTensor from '../sparseTensor/interface';
import Session from '../session/interface';

/**
 * Interface for classes that represents one of the outputs of an {@link Operation}.
 * @interface
 */
export default interface Tensor {
  /**
   * The name of the device on which this tensor will be produced, or null.
   */
  device?: string | null;

  /**
   * The {@link DType} of elements in this tensor.
   */
  dtype: DType;

  /**
   * The {@link Graph} that contains this tensor.
   */
  graph: Graph;

  /**
   * The name of this tensor.
   */
  name: string;

  /**
   * The {@link Operation} that produces this tensor as an output.
   */
  op: Operation;

  /**
   * Returns the {@link TensorShape} that represents the shape of this tensor.
   */
  shape: TensorShape;

  /**
   * The index of this tensor in the outputs of its {@link Operation}.
   */
  valueIndex: number;

  /**
   * Creates a new {@link Tensor}.
   * @param {Operation} op - {@link Operation} that computes this tensor.
   * @param {number} valueIndex - index of the operation's endpoint that produces this tensor.
   * @param {DType} dtype - type of elements stored in this tensor.
   */
  init(op: Operation, valueIndex: number, dtype: DType): void;

  /**
   * Computes the absolute value of a tensor.
   * @param {Tensor | SparseTensor} x - a {@link Tensor} or {@link SparseTensor} of type float32, float64, int32, int64,
   * complex64, complex128.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor | SparseTensor} a {@link Tensor} or {@link SparseTensor} the same size and type as x with absolute
   * values. Note, for complex64 or complex128 input, the returned {@link Tensor} will be of type float32 or float64,
   * respectively.
   */
  abs(x: Tensor | SparseTensor, name?: string): Tensor | SparseTensor;

  /**
   * Computes x + y element-wise.
   * @param {Tensor} x - a {@link Tensor}, must be one of the following types: half, bfloat16, float32, float64, uint8,
   * int8, int16, int32, int64, complex64, complex128, string.
   * @param {Tensor} y - a {@link Tensor}, must have the same type as x.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} that has the same type as x.
   */
  add(x: Tensor, y: Tensor, name?: string): Tensor;

  /**
   * Computes the truth value of x AND y element-wise.
   * @param {Tensor} x - a {@link Tensor} of type bool.
   * @param {Tensor} y - a {@link Tensor} of type bool.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of type bool.
   */
  and(x: Tensor, y: Tensor, name?: string): Tensor;

  // @TODO include method __bool__ (if needed)

  /**
   * Divide two values.
   * @param {Tensor} x - numerator of real numeric type.
   * @param {Tensor} y - denominator of real numeric type.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} quotient of x and y.
   */
  div(x: Tensor, y: Tensor, name?: string): Tensor;

  // @TODO add description for __eq__
  /**
   * @param {Tensor} other
   */
  eq(other: Tensor): boolean;

  /**
   * Divides two values element-wise, rounding toward the most negative integer.
   * @param {Tensor} x - numerator of real numeric type.
   * @param {Tensor} y - denominator of real numeric type.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} quotient of x and y rounded down.
   */
  floorDiv(x: Tensor, y: Tensor, name?: string): Tensor;

  /**
   * Computes the truth value of x >= y element-wise.
   * @param {Tensor} x - a {@link Tensor}, must be one of the following types: float32, float64, int8, int16, int32,
   * int64, uint8, uint16, uint32, uint64, bfloat16, half.
   * @param {Tensor} y - {@link Tensor}, must have the same type as x.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of type bool
   */
  ge(x: Tensor, y: Tensor, name?: string): Tensor;

  // @TODO include method __getitem__

  /**
   * Computes the truth value of x > y element-wise
   * @param {Tensor} x - a {@link Tensor}, must be one of the followint types: float32, float64, int8, int16, int32,
   * int64, uint8, uint16, uint32, uint64, bfloat16, half.
   * @param {Tensor} y - a {@link Tensor}, must have the same type as x.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of type bool.
   */
  gt(x: Tensor, y: Tensor, name?: string): Tensor;

  /**
   * Computes the trueth value of NOT x element-wise.
   * @param {Tensor} x - a {@link Tensor} of type bool.
   * @param {string} [name] - a name for the operation.
   * @return {Tensor} a {@link Tensor} of type bool.
   */
  invert(x: Tensor, name?: string): Tensor;

  // @TODO add description for __iter__
  iter(): void;

  /**
   * Computes the truth value of x <= y element-wise.
   * @param {Tensor} x - a {@link Tensor}, must be one of the following types: float32, float64, int8, int16, int32,
   * int64, uint8, uint16, uint32, uint64, bfloat16, half.
   * @param {Tensor} y - a {@link Tensor}, must have the same type as x.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of type bool.
   */
  le(x: Tensor, y: Tensor, name?: string): Tensor;

  /**
   *
   * @param {Tensor} x - a {@link Tensor}, must be one of the following types: float32, float64, int8, int16, int32,
   * int64, uint8, uint16, uint32, uint64, bfloat16, half.
   * @param {Tensor} y - a {@link Tensor}, must have the same type as x.
   * @param {string} [name] - a name for the operation
   * @returns {Tensor} a {@link Tensor} of type bool.
   */
  lt(x: Tensor, y: Tensor, name?: string): Tensor;

  /**
   * Multiplies matrix x by matrix y.
   * @param {Tensor} x - a {@link Tensor} of type float16, float32 float64, int32, complex64, complex128 and rank > 1.
   * @param {Tensor} y - a {@link Tensor} with same type and rank as x.
   * @param {bool} [transposeA] - if True, a is transposed before multiplication.
   * @param {bool} [transposeB] - if True, b is transposed before multiplication.
   * @param {bool} [adjointA] - if True, a is conjugated and transposed before multiplication.
   * @param {bool} [adjointB] - if True, b is conjugated and transposed before multiplication.
   * @param {bool} [aIsSparse] - if True, a is treated as a sparse matrix.
   * @param {bool} [bIsSparse] - if True, b is treated as a sparse matrix.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of the same type as x and y.
   */
  matMul(x: Tensor, y: Tensor, transposeA?: boolean, transposeB?: boolean, adjointA?: boolean, adjointB?: boolean,
         aIsSparse?: boolean, bIsSparse?: boolean, name?: string): Tensor;

  /**
   * Computes element-wise remainder of division.
   * @param {Tensor} x - a {@link Tensor}, must be one of the following types: int32, int64, bfloat16, float32, float64.
   * @param {Tensor} y - a {@link Tensor}, must have the same type as x.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of the same type as x.
   */
  mod(x: Tensor, y: Tensor, name?: string): Tensor;

  /**
   * @TODO add description for __mul__
   * @param {Tensor} x
   * @param {Tensor} y
   * @returns {Tensor}
   */
  mul(x: Tensor, y: Tensor): Tensor;

  /**
   * Computes numerical negative value element-wise.
   * @param {Tensor} x - a {@link Tensor}, must be one of the following types: half, bfloat16, float32, float64, int32,
   * int64, complex64, complex128.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of the same type as x.
   */
  neg(x: Tensor, name?: string): Tensor;

  // @TODO include method __nonzero__ (if needed)

  /**
   * Computes the truth value of x OR y element-wise.
   * @param {Tensor} x - a {@link Tensor} of type bool.
   * @param {Tensor} y - a {@link Tensor} of type bool.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of type bool.
   */
  or(x: Tensor, y: Tensor, name?: string): Tensor;

  /**
   * Computes the power of one value to another.
   * @param {Tensor} x - a {@link Tensor} of type float32, float64 int32, int64, complex64 or complex128.
   * @param {Tensor} y - a {@link Tensor} of type float32, float64 int32, int64, complex64 or complex128.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of the same type as x.
   */
  pow(x: Tensor, y: Tensor, name?: string): Tensor;

  /**
   * Computes x - y element-wise.
   * @param {Tensor} x - a {@link Tensor}, must be one of the following types: half, bfloat16, float32, float64, uint8,
   * uint16, int8, int16, int32, int64, complex64, complex128.
   * @param {Tensor} y - a {@link Tensor}, must have the same type as x.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of the same type as x.
   */
  sub(x: Tensor, y: Tensor, name?: string): Tensor;

  /**
   * @TODO add description for __truediv__
   * @param {Tensor} x
   * @param {Tensor} y
   * @returns {Tensor}
   */
  trueDiv(x: Tensor, y: Tensor): Tensor;

  /**
   * @TODO add description for __xor__
   * @param {Tensor} x
   * @param {Tensor} y
   * @returns {Tensor}
   */
  xor(x: Tensor, y: Tensor): Tensor;

  /**
   * Computes x + y element-wise.
   * @param {Tensor} x - a {@link Tensor}, must be one of the following types: half, bfloat16, float32, float64, uint8,
   * int8, int16, int32, int64, complex64, complex128, string.
   * @param {Tensor} y - a {@link Tensor}, must have the same type as x.
   * @param {string} [name] - a name for the operation (optional).
   * @returns {Tensor} a {@link Tensor} of the same type as x.
   */
  rAdd(x: Tensor, y: Tensor, name?: string): Tensor; // @TODO clear difference between Tensor.add

  /**
   * Computes the truth value of x AND y element-wise.
   * @param {Tensor} x - a {@link Tensor} of type bool.
   * @param {Tensor} y - a {@link Tensor} of type bool.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of type bool.
   */
  rAnd(x: Tensor, y: Tensor, name?: string): Tensor; // @TODO clear difference between Tensor.and

  /**
   * Divide two values.
   * @param {Tensor} x - numerator of real numeric type.
   * @param {Tensor} y - denominator of real numeric type.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} quotient of x and y.
   */
  rDiv(x: Tensor, y: Tensor, name?: string): Tensor; // @TODO clear difference between Tensor.div

  /**
   * Divides two values element-wise, rounding toward the most negative integer.
   * @param {Tensor} x - numerator of real numeric type.
   * @param {Tensor} y - denominator of real numeric type.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} quotient of x and y rounded down.
   */
  rFloorDiv(x: Tensor, y: Tensor, name?: string): Tensor; // @TODO clear difference between Tensor.floorDiv

  /**
   * Multiplies matrix x by matrix y.
   * @param {Tensor} x - a {@link Tensor} of type float16, float32 float64, int32, complex64, complex128 and rank > 1.
   * @param {Tensor} y - a {@link Tensor} with same type and rank as x.
   * @param {bool} [transposeA] - if True, a is transposed before multiplication.
   * @param {bool} [transposeB] - if True, b is transposed before multiplication.
   * @param {bool} [adjointA] - if True, a is conjugated and transposed before multiplication.
   * @param {bool} [adjointB] - if True, b is conjugated and transposed before multiplication.
   * @param {bool} [aIsSparse] - if True, a is treated as a sparse matrix.
   * @param {bool} [bIsSparse] - if True, b is treated as a sparse matrix.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of the same type as x and y.
   */
  rMatMul(x: Tensor, y: Tensor, transposeA?: boolean, transposeB?: boolean, adjointA?: boolean, adjointB?: boolean,
         aIsSparse?: boolean, bIsSparse?: boolean, name?: string): Tensor; // @TODO clear difference between Tensor.matMul

  /**
   * Computes element-wise remainder of division.
   * @param {Tensor} x - a {@link Tensor}, must be one of the following types: int32, int64, bfloat16, float32, float64.
   * @param {Tensor} y - a {@link Tensor}, must have the same type as x.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of the same type as x.
   */
  rMod(x: Tensor, y: Tensor, name?: string): Tensor; // @TODO clear difference between Tensor.mod

  /**
   * @TODO add description for __rmul__
   * @param {Tensor} x
   * @param {Tensor} y
   * @returns {Tensor}
   */
  rMul(x: Tensor, y: Tensor): Tensor; // @TODO clear difference between Tensor.mul

  /**
   * Computes the truth value of x OR y element-wise.
   * @param {Tensor} x - a {@link Tensor} of type bool.
   * @param {Tensor} y - a {@link Tensor} of type bool.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of type bool.
   */
  rOr(x: Tensor, y: Tensor, name?: string): Tensor; // @TODO clear difference between Tensor.or

  /**
   * Computes the power of one value to another.
   * @param {Tensor} x - a {@link Tensor} of type float32, float64 int32, int64, complex64 or complex128.
   * @param {Tensor} y - a {@link Tensor} of type float32, float64 int32, int64, complex64 or complex128.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of the same type as x.
   */
  rPow(x: Tensor, y: Tensor, name?: string): Tensor; // @TODO clear difference between Tensor.pow

  /**
   * Computes x - y element-wise.
   * @param {Tensor} x - a {@link Tensor}, must be one of the following types: half, bfloat16, float32, float64, uint8,
   * uint16, int8, int16, int32, int64, complex64, complex128.
   * @param {Tensor} y - a {@link Tensor}, must have the same type as x.
   * @param {string} [name] - a name for the operation.
   * @returns {Tensor} a {@link Tensor} of the same type as x.
   */
  rSub(x: Tensor, y: Tensor, name?: string): Tensor; // @TODO clear difference between Tensor.sub

  /**
   * @TODO add description for __rtruediv__
   * @param {Tensor} x
   * @param {Tensor} y
   * @returns {Tensor}
   */
  rTrueDiv(x: Tensor, y: Tensor): Tensor; // @TODO clear difference between Tensor.trueDiv

  /**
   * @TODO add description for __rxor__
   * @param {Tensor} x
   * @param {Tensor} y
   * @returns {Tensor}
   */
  rXor(x: Tensor, y: Tensor): Tensor; // @TODO clear difference between Tensor.xor

  /**
   * Returns a list of {@link Operation}s that consume this tensor.
   * @returns {Operation[]}
   */
  consumers(): Operation[];

  /**
   * Evaluates this tensor in a {@link Session}.
   * @param {Tensor[]} feedDict - a dictionary that maps {@link Tensor} objects to feed values.
   * @param {Session} [session] - a {@link Session} to be used to evaluate this tensor. If none, the default session
   * will be used.
   * @returns {Array} - an array corresponding to the value of this tensor.
   */
  eval(feedDict: Tensor[], session?: Session): Tensor[]; // @TODO clear this method

  /**
   * Gets the shape of this tensor
   * @returns {TensorShape}
   */
  getShape(): TensorShape;

  /**
   * Updates the shape of this tensor
   * @param {TensorShape} shape - a {@link TensorShape} representing the shape of this tensor.
   */
  setShape(shape: TensorShape): void;
}
