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

package org.apache.spark.ml.regression

import org.apache.spark.ml.param.{ ParamMap, Params, DoubleParam, IntParam, ParamValidators }

// TODO - Move into org.apache.spark.ml.param.shared.SharedParamsCodeGen.
private[ml] trait HasLambdaIndex extends Params {

  /**
   * Param for lambda index (>= 0).
   * @group param
   */
  final val lambdaIndex: IntParam = new IntParam(this, "lambdaIndex", "lambda index (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  final def getLambdaIndex: Int = $(lambdaIndex)
}

private[ml] trait HasNumLambdas extends Params {
  /**
   * Param for lambda index (>= 0).
   * @group param
   */
  final val numLambdas: IntParam = new IntParam(this, "numLambdas", "number of lambdas (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  final def getNumLambdas: Int = $(numLambdas)
}

private[ml] trait HasLambdaShrink extends Params {
  /**
   * Param for lambda index (>= 0).
   * @group param
   */
  final val lambdaShrink: DoubleParam = new DoubleParam(this, "lambdaShrink", "lambda shrink (>= 0.0)", ParamValidators.gtEq(0))

  /** @group getParam */
  final def getLambdaShrink: Double = $(lambdaShrink)
}

private[ml] trait HasNumFolds extends Params {
  /**
   * Param for lambda index (>= 0).
   * @group param
   */
  final val numFolds: IntParam = new IntParam(this, "numFolds", "numFolds (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  final def getNumFolds: Int = $(numFolds)
}