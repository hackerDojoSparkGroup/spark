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

package org.apache.spark.mllib.stat

import org.apache.spark.mllib.linalg.Matrix

class MatrixMultivariateOnlineSummarizer extends MultivariateOnlineSummarizer_Modified {

  /**
   * Add a new sample to this summarizer, and update the statistical summary.
   *
   * @param sample The sample in dense/sparse matrix format to be added into this summarizer.
   * @return This MatrixMultivariateOnlineSummarizer object.
   */
  def add(sample: Matrix): this.type = {
    if (n == 0) {
      require(sample.numRows * sample.numCols > 0, s"Matrix should have dimension larger than zero.")
      n = sample.numCols

      currMean = Array.ofDim[Double](n)
      currM2n = Array.ofDim[Double](n)
      currM2 = Array.ofDim[Double](n)
      currL1 = Array.ofDim[Double](n)
      nnz = Array.ofDim[Double](n)
      currMax = Array.fill[Double](n)(Double.MinValue)
      currMin = Array.fill[Double](n)(Double.MaxValue)
    }

    require(n == sample.numCols, s"Dimensions mismatch when adding new sample." +
      s" Expecting $n columns but got ${sample.numCols}.")

    val localCurrMean = currMean
    val localCurrM2n = currM2n
    val localCurrM2 = currM2
    val localCurrL1 = currL1
    val localNnz = nnz
    val localCurrMax = currMax
    val localCurrMin = currMin
    sample.foreachActive { (_, index, value) =>
      if (value != 0.0) {
        if (localCurrMax(index) < value) {
          localCurrMax(index) = value
        }
        if (localCurrMin(index) > value) {
          localCurrMin(index) = value
        }

        val prevMean = localCurrMean(index)
        val diff = value - prevMean
        localCurrMean(index) = prevMean + diff / (localNnz(index) + 1.0)
        localCurrM2n(index) += (value - localCurrMean(index)) * diff
        localCurrM2(index) += value * value
        localCurrL1(index) += math.abs(value)

        localNnz(index) += 1.0
      }
    }

    totalCnt += sample.numRows
    this
  }
}