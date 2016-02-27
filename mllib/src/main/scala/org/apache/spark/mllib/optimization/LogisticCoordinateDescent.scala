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

package org.apache.spark.mllib.optimization

import breeze.linalg.{ DenseMatrix, DenseVector }
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.Logging
import org.apache.spark.ml.classification.Stats3
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd._
import scala.collection.mutable.MutableList
import scala.math.{ abs, exp, sqrt }
import scala.annotation.tailrec
import TempTestUtil.verifyResults

private[spark] class LogisticCoordinateDescent extends CoordinateDescentParams
  with Logging {

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, stats: Stats3, numRows: Long): List[(Double, Vector)] = {
    LogisticCoordinateDescent.runCD(
      data,
      initialWeights,
      elasticNetParam,
      lambdaShrink,
      numLambdas,
      maxIter,
      tol,
      stats,
      numRows)
  }
}

/**
 * :: DeveloperApi ::
 * Top-level method to run coordinate descent.
 */
@DeveloperApi
object LogisticCoordinateDescent extends Logging {

  def runCD(data: RDD[(Double, Vector)], initialWeights: Vector, alpha: Double, lamShrnk: Double, numLambdas: Int, maxIter: Int, tol: Double, stats: Stats3, numRows: Long): List[(Double, Vector)] = {
    logInfo(s"Performing coordinate descent with: [elasticNetParam: $alpha, lamShrnk: $lamShrnk, numLambdas: $numLambdas, maxIter: $maxIter, tol: $tol]")

    val lamMult = 0.93

    val (lambdas, initialBeta0) = computeLambdasAndInitialBeta0(data, alpha, lamMult, numLambdas, stats, numRows)
    val lambdasAndBetas = optimize(data, lambdas, initialBeta0, alpha, stats, numRows)

    //TODO - Return the column order and put that into the model as part of the history. Or better yet, 
    // columnOrder should be calculated in the example code from the List of models containing the beta history using a util class
    val columnOrder = determineColumnOrder(lambdasAndBetas.unzip._2)
    lambdasAndBetas
  }

  private def computeLambdasAndInitialBeta0(data: RDD[(Double, Vector)], alpha: Double, lambdaMult: Double, numLambdas: Int, stats: Stats3, numRows: Long): (Array[Double], Double) = {
    //logDebug(s"alpha: $alpha, lamShrnk: $lamShrnk, maxIter: $lambdaRange, numRows: $numRows")

    val p = stats.yMean
    val w = p * (1.0 - p)
    val sumW = w * numRows

    val initialWeights = data.treeAggregate(new ComputeInitialWeights(p, stats.numFeatures))(
      (aggregate, row) => aggregate.compute(row),
      (aggregate1, aggregate2) => aggregate1.combine(aggregate2))

    val sumWxr = initialWeights.sumWxr
    val sumWr = initialWeights.sumWr

    val avgWxr = sumWxr.toBreeze / numRows.toDouble
    val maxWxr = avgWxr.map(abs).max(Ordering.Double)

    // calculate starting value for lambda
    val lamdaInit = maxWxr / alpha

    // the lamdaInit value of lambda corresponds to a beta of all 0's
    val beta0 = sumWr / sumW

    val lambdas = Array.iterate[Double](lamdaInit * lambdaMult, numLambdas)(_ * lambdaMult)
    (lambdas, beta0)
  }

  private def optimize(data: RDD[(Double, Vector)], lambdas: Array[Double], initialBeta0: Double, alpha: Double, stats: Stats3, numRows: Long): List[(Double, Vector)] = {
    //initial value of lambda corresponds to beta = list of 0's
    //TODO - Convert beta to Vector
    var beta = Array.ofDim[Double](stats.numFeatures)
    val beta0 = initialBeta0

    val betaMat = MutableList.empty[Array[Double]]
    //TODO - Do not return the initial beta value of all zero's, only return a list of 100
    betaMat += beta.clone

    //TODO - beta0 does not change value, so no need to collect it in a list    
    val beta0List = MutableList.empty[Double]
    beta0List += beta0

    val nzList = MutableList.empty[Int]

    loop(beta, 0)

    /*loop to decrement lambda and perform iteration for betas*/
    @tailrec
    def loop(oldBeta: Array[Double], n: Int): Unit = {
      if (n < lambdas.length) {
        val newLambda = lambdas(n)
        val newBeta = outerLoop(n + 1, data, oldBeta, beta0, newLambda, alpha, stats.numFeatures, numRows)
        betaMat += newBeta.clone
        beta0List += beta0
        loop(newBeta, n + 1)
      }
    }

    verifyResults(stats, stats.yMean, stats.yStd, betaMat, beta0List)

    //TODO - Do not combine beta0 with the other betas, return it as a separate member of the tuples -> List[(Double, Double, Vector)]
    val fullBetas = beta0List.zip(betaMat).map { case (b0, beta) => Vectors.dense(b0 +: beta) }
    lambdas.zip(fullBetas).toList
  }

  private def outerLoop(iStep: Int, data: RDD[(Double, Vector)], oldBeta: Array[Double], beta0: Double, lambda: Double, alpha: Double, numFeatures: Int, numRows: Long): Array[Double] = {
    //Use incremental change in betas to control inner iteration
    //set middle loop values for betas = to outer values
    // values are used for calculating weights and probabilities
    //inner values are used for calculating penalized regression updates
    //take pass through data to calculate averages over data require for iteration
    //initialize accumulators

    val outer = data.treeAggregate(new CalculateOuter(beta0, oldBeta, numFeatures))(
      (aggregate, row) => aggregate.compute(row),
      (aggregate1, aggregate2) => aggregate1.combine(aggregate2))

    val wXX = outer.wXX
    val wX = outer.wX
    val wXz = outer.wXz
    val wZ = outer.wZ
    val wSum = outer.wSum

    def loop(iterIRLS: Int, betaIRLS: Array[Double], distIRLS: Double): Array[Double] = {
      if (distIRLS <= 0.01) betaIRLS
      else {
        val (newBetaIRLS, newDistIRLS) = middleLoop(iStep, iterIRLS, betaIRLS, beta0, wXX, wX, wXz, wZ, wSum, lambda, alpha, numFeatures, numRows)
        loop(0, newBetaIRLS, newDistIRLS)
      }
    }

    loop(0, oldBeta, 100.0)
  }

  private def middleLoop(iStep: Int, iterIRLS: Int, betaIRLS: Array[Double], beta0IRLS: Double, wXX: DenseMatrix[Double], wX: DenseVector[Double], wXz: DenseVector[Double], wZ: Double, wSum: Double, lambda: Double, alpha: Double, numColumns: Int, numRows: Long): (Array[Double], Double) = {
    @tailrec
    def loop(iterInner: Int, distInner: Double, oldBeta0Inner: Double, mutableBetaInner: DenseVector[Double]): (Int, Array[Double]) = {
      if (iterInner >= 100 || distInner <= 0.01) (iterInner, mutableBetaInner.toArray)
      else {
        val (newDistInner, newBeta0Inner) = innerLoop(mutableBetaInner, oldBeta0Inner, wXX, wX, wXz, wZ, wSum, lambda, alpha, numColumns, numRows)
        loop(iterInner + 1, newDistInner, newBeta0Inner, mutableBetaInner)
      }
    }

    val (iterInner, betaInner) = loop(0, 100.0, beta0IRLS, DenseVector(betaIRLS.clone))

    println(iStep, iterIRLS, iterInner)

    //Check change in betaMiddle to see if IRLS is converged
    val a = (for (i <- 0 until numColumns) yield (abs(betaIRLS(i) - betaInner(i)))).sum
    val b = (for (i <- 0 until numColumns) yield abs(betaIRLS(i))).sum
    val distIRLS = a / (b + 0.0001)
    val dBeta = for (i <- 0 until numColumns) yield (betaInner(i) - betaIRLS(i))
    val newBetaIRLS = for (i <- 0 until numColumns) yield (betaIRLS(i) + dBeta(i))
    (newBetaIRLS.toArray, distIRLS)
  }

  /** The betaInner input parameter will be mutated. */
  private def innerLoop(betaInner: DenseVector[Double], oldBeta0Inner: Double, wXX: DenseMatrix[Double], wX: DenseVector[Double], wXz: DenseVector[Double], wZ: Double, wSum: Double, lambda: Double, alpha: Double, numColumns: Int, numRows: Long): (Double, Double) = {
    var beta0Inner = oldBeta0Inner

    //cycle through attributes and update one-at-a-time
    //record starting value for comparison
    val betaStart = betaInner.toArray.clone
    for (iCol <- 0 until numColumns) {
      val sumWxrC = wXz(iCol) - wX(iCol) * beta0Inner - (wXX(::, iCol) dot betaInner)
      val sumWrC = wZ - wSum * beta0Inner - (wX dot betaInner)

      val avgWxr = sumWxrC / numRows
      val avgWxx = wXX(iCol, iCol) / numRows

      beta0Inner = beta0Inner + sumWrC / wSum
      val uncBeta = avgWxr + avgWxx * betaInner(iCol)
      betaInner(iCol) = S(uncBeta, lambda * alpha) / (avgWxx + lambda * (1.0 - alpha))
    }
    val sumDiff = (for (n <- 0 until numColumns) yield (abs(betaInner(n) - betaStart(n)))).sum
    val sumBeta = (for (n <- 0 until numColumns) yield abs(betaInner(n))).sum
    val distInner = sumDiff / sumBeta

    (distInner, beta0Inner)
  }

  private def S(z: Double, gamma: Double): Double =
    if (gamma >= abs(z)) 0.0
    else if (z > 0.0) z - gamma
    else z + gamma

  private def determineColumnOrder(betas: List[Vector]): Array[Int] = {
    val nzList = betas
      .map(_.toArray.drop(1).zipWithIndex.filter(_._1 != 0.0).map(_._2))
      .flatMap(f => f)
      .distinct

    //make up names for columns of xNum
    val nameList = nzList.map(index => s"V$index")

    println(nameList)
    verifyResults(nameList)

    nzList.toArray
  }
}

private class ComputeInitialWeights(p: Double, numFeatures: Int) extends Serializable {

  lazy val sumWxr = Vectors.zeros(numFeatures)
  var sumWr = 0.0

  def compute(row: (Double, Vector)): this.type = {
    val y = row._1
    val x = row._2
    val wr = y - p
    BLAS.axpy(wr, x, sumWxr)
    sumWr += wr
    this
  }

  def combine(other: ComputeInitialWeights): this.type = {
    sumWxr.toBreeze :+= other.sumWxr.toBreeze
    sumWr += other.sumWr
    this
  }
}

//TODO - Convert beta to Vector and broadcast beta
//TODO - Improve efficiency of operations in CalculateOuter
private class CalculateOuter(beta0: Double, beta: Array[Double], numFeatures: Int) extends Serializable {

  lazy val wXX = DenseMatrix.zeros[Double](numFeatures, numFeatures)
  lazy val wX = DenseVector.zeros[Double](numFeatures)
  lazy val wXz = DenseVector.zeros[Double](numFeatures)
  var wZ = 0.0
  var wSum = 0.0

  def compute(row: (Double, Vector)): this.type = {
    val y = row._1
    val x = row._2.toArray
    val (p, w) = adjPW(beta0, beta, x)
    val xNP = DenseVector(x)
    wXX += w * (xNP * xNP.t)
    wX += w * xNP
    // residual for logistic
    val z = (y - p) / w + beta0 + (for (i <- 0 until numFeatures) yield (x(i) * beta(i))).sum
    wXz += w * xNP * z
    wZ += w * z
    wSum += w
    this
  }

  def combine(other: CalculateOuter): this.type = {
    wXX :+= other.wXX
    wX :+= other.wX
    wXz :+= other.wXz
    wZ += other.wZ
    wSum += other.wSum
    this
  }

  // performs adjustments recommended by Friedman for numerical stability
  private def adjPW(b0: Double, b: Array[Double], x: Array[Double]): (Double, Double) = {
    val pr = Pr(b0, b, x)
    if (abs(pr) < 1e-5) (0.0, 1e-5)
    else if (abs(1.0 - pr) < 1e-5) (1.0, 1e-5)
    else (pr, pr * (1.0 - pr))
  }

  private def Pr(b0: Double, b: Array[Double], x: Array[Double]): Double = {
    val n = x.length
    var sum = b0
    var j = 0
    while (j < n) {
      sum += b(j) * x(j)
      sum = if (sum < -100) -100 else sum
      j += 1
    }
    1.0 / (1.0 + exp(-sum))
  }
}  
