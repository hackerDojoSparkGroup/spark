# Spark-GLMNET

This branch contains implementations of the algorithm first described in “Regularization Paths for Generalized Linear Models via Coordinate Descent” algorithm by Jerome Friedman, Trevor Hastie and Rob Tibshirani of Stanford University (http://web.stanford.edu/~hastie/Papers/glmnet.pdf). The algorithm is typically referred to as “glmnet” - generalized linear model with elastic net regularization. Elastic net is a penalized regression method.  The elastic net penalty is a combination of the ridge and lasso penalties. Penalized regression has several benefits.  It will generate useful solutions for problems that are under-determined, for example problems involving genetic data and penalized regression yields variable importance information, which is helpful in early stages of problem formulation. 

Glmnet does much more than generate point solutions to the penalized regression problem.  It solves the penalized regression problem for a wide range of values of the regularization parameter and thus generates "regularization paths" referred to in the title of Friedman, Hastie and Tibshirani's paper.  This makes glmnet well-suited to the pipeline architecture now available in Spark.  

The goal of our Spark implementation has been to produce a package very similar in functionality to the cran-R package "glmnet" that was authored by paper's authors.  Like the cran-R package, this Spark implementation provides both regression and binary classification versions, generates regularization curves and if requested, will makes cross-validation runs and select the value of the regularization parameter that gives the best cross-validated error (in several senses).  Like the cran-R package, this Spark implementation only requires the data matrix along with appropriate labels and a few choices.  It usually gives meaningful answers with default settings and when it doesn't the required changes are clear.  We have hewn as close as possible to the algorithm described in the original paper in hopes of matching glmnet's ease of use on datasets that won't fit or are to slow to train with cran-R or python versions of the algorithm.  

## Building Spark

Spark is built using [Apache Maven](http://maven.apache.org/).
To build Spark and its example programs, run:

    export MAVEN_OPTS="-Xmx2g -XX:MaxPermSize=512M -XX:ReservedCodeCacheSize=512m"
    build/mvn -DskipTests clean package

## Example Programs

There are three sample programs for Spark-GLMNET in the `examples/ml` directory.
* LinearRegressionWithCDExample - Produces a single model using linear regression with coordinate descent
* LogisticRegressionWithCDExample - Produces a single model using logistic regression with coordinate descent
* LinearRegressionWithCDCrossValidatorExample - Produces a best-fit model using cross-validation with auto-generated lambda parameter values and linear regression with coordinate descent.

To run one of them, use `./bin/run-example <class> [params]`. For example:

    ./bin/run-example ml.LinearRegressionWithCDExample

will run the LinearRegressionWithCDExample example locally.

You can set the MASTER environment variable when running examples to submit
examples to a cluster. This can be a mesos:// or spark:// URL,
"yarn" to run on YARN, and "local" to run
locally with one thread, or "local[N]" to run locally with N threads. You
can also use an abbreviated class name if the class is in the `examples`
package. For instance:

    MASTER=spark://host:7077 ./bin/run-example SparkPi

Many of the example programs print usage help if no params are given.


# Apache Spark

Spark is a fast and general cluster computing system for Big Data. It provides
high-level APIs in Scala, Java, Python, and R, and an optimized engine that
supports general computation graphs for data analysis. It also supports a
rich set of higher-level tools including Spark SQL for SQL and DataFrames,
MLlib for machine learning, GraphX for graph processing,
and Spark Streaming for stream processing.

<http://spark.apache.org/>


## Online Documentation

You can find the latest Spark documentation, including a programming
guide, on the [project web page](http://spark.apache.org/documentation.html)
and [project wiki](https://cwiki.apache.org/confluence/display/SPARK).
This README file only contains basic setup instructions.

## Building Spark

Spark is built using [Apache Maven](http://maven.apache.org/).
To build Spark and its example programs, run:

    build/mvn -DskipTests clean package

(You do not need to do this if you downloaded a pre-built package.)
More detailed documentation is available from the project site, at
["Building Spark"](http://spark.apache.org/docs/latest/building-spark.html).
For developing Spark using an IDE, see [Eclipse](https://cwiki.apache.org/confluence/display/SPARK/Useful+Developer+Tools#UsefulDeveloperTools-Eclipse)
and [IntelliJ](https://cwiki.apache.org/confluence/display/SPARK/Useful+Developer+Tools#UsefulDeveloperTools-IntelliJ).

## Interactive Scala Shell

The easiest way to start using Spark is through the Scala shell:

    ./bin/spark-shell

Try the following command, which should return 1000:

    scala> sc.parallelize(1 to 1000).count()

## Interactive Python Shell

Alternatively, if you prefer Python, you can use the Python shell:

    ./bin/pyspark

And run the following command, which should also return 1000:

    >>> sc.parallelize(range(1000)).count()

## Example Programs

Spark also comes with several sample programs in the `examples` directory.
To run one of them, use `./bin/run-example <class> [params]`. For example:

    ./bin/run-example SparkPi

will run the Pi example locally.

You can set the MASTER environment variable when running examples to submit
examples to a cluster. This can be a mesos:// or spark:// URL,
"yarn" to run on YARN, and "local" to run
locally with one thread, or "local[N]" to run locally with N threads. You
can also use an abbreviated class name if the class is in the `examples`
package. For instance:

    MASTER=spark://host:7077 ./bin/run-example SparkPi

Many of the example programs print usage help if no params are given.

## Running Tests

Testing first requires [building Spark](#building-spark). Once Spark is built, tests
can be run using:

    ./dev/run-tests

Please see the guidance on how to
[run tests for a module, or individual tests](https://cwiki.apache.org/confluence/display/SPARK/Useful+Developer+Tools).

## A Note About Hadoop Versions

Spark uses the Hadoop core library to talk to HDFS and other Hadoop-supported
storage systems. Because the protocols have changed in different versions of
Hadoop, you must build Spark against the same version that your cluster runs.

Please refer to the build documentation at
["Specifying the Hadoop Version"](http://spark.apache.org/docs/latest/building-spark.html#specifying-the-hadoop-version)
for detailed guidance on building for a particular distribution of Hadoop, including
building for particular Hive and Hive Thriftserver distributions.

## Configuration

Please refer to the [Configuration Guide](http://spark.apache.org/docs/latest/configuration.html)
in the online documentation for an overview on how to configure Spark.
