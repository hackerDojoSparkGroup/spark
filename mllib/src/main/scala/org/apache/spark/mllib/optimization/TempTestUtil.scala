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

import org.apache.spark.ml.classification.Stats3
import scala.collection.mutable.MutableList
import scala.io.Source
import scala.math.abs

object TempTestUtil {

  def verifyResults(stats: Stats3, meanLabel: Double, sdLabel: Double, betaMat: MutableList[Array[Double]], beta0List: MutableList[Double]) = {
    val tolerance = 1e-12

    val expectedXmeans = FileUtil.readFile("results/logistic-regression/xMeans.txt")(0)
      .split(",").map(_.toDouble).toArray
    TestUtil.equalWithinTolerance(stats.featuresMean.toArray, expectedXmeans, tolerance, "xMeans")

    val expectedXSD = FileUtil.readFile("results/logistic-regression/xSDwithBesselsCorrection.txt")(0)
      .split(",").map(_.toDouble).toArray
    TestUtil.equalWithinTolerance(stats.featuresStd.toArray, expectedXSD, tolerance, "xSD")

    val expectedYmean = FileUtil.readFile("results/logistic-regression/yMean.txt")(0).toDouble
    TestUtil.equalWithinTolerance(meanLabel, expectedYmean, tolerance, "yMean")

    val expectedYSD = FileUtil.readFile("results/logistic-regression/ySDwithBesselsCorrection.txt")(0).toDouble
    TestUtil.equalWithinTolerance(sdLabel, expectedYSD, tolerance, "ySD")

    val expectedBetaMat = FileUtil.readFile("results/logistic-regression/betaMatWithBesselsCorrection.txt")
      .map(_.split(",").map(_.toDouble)).toArray
    TestUtil.equalWithinTolerance(betaMat.toArray, expectedBetaMat, tolerance, "betas")

    val expectedBeta0List = FileUtil.readFile("results/logistic-regression/beta0List.txt")(0)
      .split(",").map(_.toDouble).toArray
    TestUtil.equalWithinTolerance(beta0List.toArray, expectedBeta0List, tolerance, "beta0s")

    println("\nResults Verified\n")
  }

  def verifyResults(nameList: List[String]) = {
    val expectedNamelist = FileUtil.readFile("results/logistic-regression/namelist.txt")(0)
      .split(",").map(_.trim).toArray
    TestUtil.equal(nameList.toArray, expectedNamelist, "columnOrder")

    println("\nnameList Verified\n")
  }
}

object TestUtil {

  def equalWithinTolerance(actual: Array[Array[Double]], expected: Array[Array[Double]], tolerance: Double, testName: String): Unit = {
    if (actual.length != expected.length)
      sys.error(s"$testName: The actual number of rows ${actual.length} do not match the expected number of rows ${expected.length}")
    actual.zip(expected).zipWithIndex.foreach {
      case ((a, e), row) => equalWithinTolerance(a, e, tolerance, testName)
    }
  }

  def equalWithinTolerance(actual: Array[Double], expected: Array[Double], tolerance: Double, testName: String): Unit = {
    if (actual.length != expected.length)
      sys.error(s"$testName: The actual number of columns ${actual.length} do not match the expected number of columns ${expected.length}")
    actual.zip(expected).zipWithIndex.foreach {
      case ((a, e), column) => equalWithinTolerance(a, e, tolerance, testName)
    }
  }

  def equalWithinTolerance(actual: Double, expected: Double, tolerance: Double, testName: String): Unit =
    if (abs(expected - actual) > tolerance)
      sys.error(s"$testName: The difference between the expected [$expected] and actual [$actual] value is not within the tolerance of [$tolerance]")

  def equal(actual: Array[String], expected: Array[String], testName: String): Unit =
    actual.zip(expected).foreach {
      case (a, e) => if (a != e) sys.error(s"The actual [$a] is not equal to the expected [$e] value")
    }
}

object FileUtil {
  def readFile(filename: String): List[String] = {
    val bufferedSource = Source.fromFile(filename)
    val lines = bufferedSource.getLines.toList
    bufferedSource.close
    lines
  }
}
