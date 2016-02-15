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

// scalastyle:off println
package org.apache.spark.examples.ml

import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.ml.classification.LogisticRegressionWithCD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

object LogisticRegressionWithCDExample {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("LogisticRegressionWithCDExample")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val fileReader = FileUtil.readLabeledPoints(sc, ',', false, label => if (label == "M") 1.0 else 0.0)_
    val training = fileReader("data/mllib/sonar.all-data")

    val lr = new LogisticRegressionWithCD("")
    lr.setMaxIter(100)
      .setElasticNetParam(1.0)

    val model = lr.fit(training.toDF())

    sc.stop()
  }
}

object FileUtil {

  def readLabeledPoints(sc: SparkContext, delimiter: Char, labelInFirstColumn: Boolean, labelConversion: String => Double)(path: String): RDD[LabeledPoint] = {
    readLabeledPoints(sc.textFile(path), delimiter, labelInFirstColumn, labelConversion)
  }

  def readLabeledPoints(sc: SparkContext, delimiter: Char, labelInFirstColumn: Boolean, minPartitions: Int, labelConversion: String => Double)(path: String): RDD[LabeledPoint] = {
    val rdd = if (minPartitions < 1) sc.textFile(path) else sc.textFile(path, minPartitions)
    readLabeledPoints(rdd, delimiter, labelInFirstColumn, labelConversion)
  }

  private def readLabeledPoints(fileRDD: RDD[String], delimiter: Char, labelInFirstColumn: Boolean, labelConversion: String => Double): RDD[LabeledPoint] = {
    val extractLabel = if (labelInFirstColumn) (row: Array[String]) => row.take(1)(0)
    else (row: Array[String]) => row.drop(row.size - 1)(0)

    val extractFeatures = if (labelInFirstColumn) (row: Array[String]) => row.drop(1)
    else (row: Array[String]) => row.take(row.size - 1)

    fileRDD
      .map(_.split(delimiter))
      .map(row => LabeledPoint(labelConversion(extractLabel(row)), Vectors.dense(extractFeatures(row).map(_.toDouble))))
  }
}
// scalastyle:on println
