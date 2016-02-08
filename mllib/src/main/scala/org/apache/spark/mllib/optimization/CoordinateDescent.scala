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

import org.apache.spark.ml.param.{ ParamMap, Params, IntParam, ParamValidators }
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.PredictorParams

/**
 * Params for coordinate descent.
 */
private[spark] trait CoordinateDescentParams {
  var elasticNetParam: Double = 0.01
  var numLambdas: Int = 100
  var lambdaShrink: Double = 0.001
  var maxIter: Int = 100
  var tol: Double = 1E-3
  var logSaveAll: Boolean = false

  /**
   * Set the elasticNetParam. Default 0.01.
   */
  def setElasticNetParam(elasticNetParam: Double): this.type = {
    this.elasticNetParam = elasticNetParam
    this
  }

    /**
   * Set the number of lambdas for CD. Default 100.
   */
  def setNumLambdas(numLambdas: Int): this.type = {
    this.numLambdas = numLambdas
    this
  }

  /**
   * Set the lambda shrinkage parameter. Default 0.001.
   */
  def setLambdaShrink(lambdaShrink: Double): this.type = {
    this.lambdaShrink = lambdaShrink
    this
  }

  /**
   * Set the number of iterations for CD. Default 100.
   */
  def setMaxIter(maxIter: Int): this.type = {
    this.maxIter = maxIter
    this
  }

  /**
   * Set the tol. Default 0.01.
   */
  def setTol(tol: Double): this.type = {
    this.tol = tol
    this
  }

  /**
   * Set logSaveAll. Default false.
   */
  def setLogSaveAll(logSaveAll: Boolean): this.type = {
    this.logSaveAll = logSaveAll
    this
  }
}
