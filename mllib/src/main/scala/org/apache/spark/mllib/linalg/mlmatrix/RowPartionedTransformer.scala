/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.linalg.mlmatrix

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{ Vector, DenseMatrix, DenseVector }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ DataFrame, Row }

// Adapted from https://github.com/amplab/ml-matrix/blob/master/src/main/scala/edu/berkeley/cs/amplab/mlmatrix/RowPartitionedMatrix.scala
//TODO - Investigate using org.apache.spark.mllib.linalg.distributed.BlockMatrix instead of this
//TODO - Investigate usage as a org.apache.spark.ml.Transformer

/** Note: [[breeze.linalg.DenseMatrix]] by default uses column-major layout. */
/**
 * Transforms RDD's into a single row per partition, allowing higher level BLAS operations to operate on the RDD contents.
 *  For example, Doubles can be converted to a DenseVector and Array[Double]s or DenseVectors can be converted to a DenseMatrix.
 */
object RowPartionedTransformer extends Logging {

  def labeledPointsToMatrix(dataset: DataFrame): RDD[(DenseVector, DenseMatrix)] = {
    val rowRDD = dataset.select("label", "features").map {
      case Row(label: Double, features: Vector) =>
        (label, features.toArray)
    }

    val rowsColsPerPartition = rowRDD.mapPartitionsWithIndex {
      case (part, iter) =>
        if (iter.hasNext) {
          val nCols = iter.next()._2.size
          Iterator((part, 1 + iter.size, nCols))
        } else {
          Iterator((part, 0, 0))
        }
    }.collect().sortBy(x => (x._1, x._2, x._3)).map(x => (x._1, (x._2, x._3))).toMap

    logDebug(s"rowsColsPerPartition: $rowsColsPerPartition")
    val rBroadcast = rowRDD.context.broadcast(rowsColsPerPartition)

    val data = rowRDD.mapPartitionsWithIndex {
      case (part, iter) =>
        val (rows, cols) = rBroadcast.value(part)
        val vecData = new Array[Double](rows)
        val matData = new Array[Double](rows * cols)
        var nRow = 0
        while (iter.hasNext) {
          val row = iter.next()
          vecData(nRow) = row._1
          val arr = row._2
          var idx = 0
          while (idx < arr.size) {
            matData(nRow + idx * rows) = arr(idx)
            idx = idx + 1
          }
          nRow += 1
        }
        Iterator((new DenseVector(vecData), new DenseMatrix(rows, cols, matData)))
    }
    data
  }

  def arrayToMatrix(matrixRDD: RDD[Array[Double]]): RDD[DenseMatrix] = {
    val rowsColsPerPartition = matrixRDD.mapPartitionsWithIndex {
      case (part, iter) =>
        if (iter.hasNext) {
          val nCols = iter.next().size
          Iterator((part, 1 + iter.size, nCols))
        } else {
          Iterator((part, 0, 0))
        }
    }.collect().sortBy(x => (x._1, x._2, x._3)).map(x => (x._1, (x._2, x._3))).toMap

    logDebug(s"rowsColsPerPartition: $rowsColsPerPartition")
    val rBroadcast = matrixRDD.context.broadcast(rowsColsPerPartition)

    val data = matrixRDD.mapPartitionsWithIndex {
      case (part, iter) =>
        val (rows, cols) = rBroadcast.value(part)
        val matData = new Array[Double](rows * cols)
        var nRow = 0
        while (iter.hasNext) {
          val arr = iter.next()
          var idx = 0
          while (idx < arr.size) {
            matData(nRow + idx * rows) = arr(idx)
            idx = idx + 1
          }
          nRow += 1
        }
        Iterator(new DenseMatrix(rows, cols, matData.toArray))
    }
    data
  }
}
