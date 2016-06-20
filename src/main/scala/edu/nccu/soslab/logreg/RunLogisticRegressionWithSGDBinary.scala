package edu.nccu.soslab.logreg

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.joda.time._
import org.jfree.data.category.DefaultCategoryDataset
import org.apache.spark.mllib.linalg.Vector
import org.apache.log4j.BasicConfigurator
import org.apache.commons.cli.Options
import org.apache.commons.cli.CommandLineParser
import org.apache.commons.cli.ParseException
import org.apache.commons.cli.CommandLine
import org.apache.commons.cli.HelpFormatter
import org.apache.commons.cli.PosixParser
import org.apache.hadoop.fs.Path

object RunLogisticRegressionWithSGDBinary {

	val logger = Logger.getLogger(RunLogisticRegressionWithSGDBinary.getClass);

	def main(args: Array[String]) {

		BasicConfigurator.configure()

		val options = new Options()
		options.addOption("s", "single-node", false, "是否為單機模式");
		options.addOption("t", "param-tune", false, "進行參數調校");
		options.addOption("i", "input-dir", true, "輸入資料夾");
		options.addOption("o", "output-dir", true, "輸出資料夾");

		val cmdParser = new PosixParser();
		var cmdLine: CommandLine = null;
		try {
			cmdLine = cmdParser.parse(options, args);
		}
		catch {
			case ex: ParseException => {
				printHelp(options);
				System.exit(1);
			}
		}

		if (!cmdLine.hasOption('i')) {
			logger.fatal("沒有提供輸入資料夾");
			printHelp(options);
			System.exit(1);
		}
		if (!cmdLine.hasOption('o')) {
			logger.fatal("沒有提供輸出資料夾");
			printHelp(options);
			System.exit(1);
		}

		val trainTSV = new Path(cmdLine.getOptionValue('i'), "train.tsv");
		val testTSV = new Path(cmdLine.getOptionValue('i'), "test.tsv");
		val outputDir = new Path(cmdLine.getOptionValue('o'));

		logger.info("train.tsv 位置:" + trainTSV.toString())
		logger.info("test.tsv 位置:" + testTSV.toString())
		logger.info("輸出資料夾位置:" + outputDir)

		val sparkConf = new SparkConf().setAppName("RDF");

		if (cmdLine.hasOption('s')) {
			sparkConf.setMaster("local[4]")
		}

		val sc = new SparkContext(sparkConf)
		logger.info("RunLogisticRegressionWithSGDBinary")
		logger.info("==========資料準備階段===============")
		val (trainData, validationData, testData, categoriesMap) = PrepareData(sc, trainTSV)
		trainData.persist(); validationData.persist(); testData.persist()

		if (cmdLine.hasOption('t')) {
			val model = parametersTunning(trainData, validationData)
			logger.info("==========測試階段===============")
			val auc = evaluateModel(model, testData)
			logger.info("使用testata測試最佳模型,結果 AUC:" + auc)
			logger.info("==========預測資料===============")
			PredictData(sc, model, categoriesMap, testTSV, outputDir)
		}
		else {
			val model = trainEvaluate(trainData, validationData)
			logger.info("==========測試階段===============")
			val auc = evaluateModel(model, testData)
			logger.info("使用testata測試模型,結果 AUC:" + auc)
			logger.info("==========預測資料===============")
			PredictData(sc, model, categoriesMap, testTSV, outputDir)
		}

		trainData.unpersist(); validationData.unpersist(); testData.unpersist()

	}

	def PrepareData(sc: SparkContext, dataTSVPath: Path): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint], Map[String, Int]) = {
		//----------------------1.匯入轉換資料-------------
		logger.info("開始匯入資料...")
		val rawDataWithHeader: RDD[String] = sc.textFile(dataTSVPath.toString())
		val rawData: RDD[String] = rawDataWithHeader.mapPartitionsWithIndex {
			(idx, iter) => if (idx == 0) iter.drop(1) else iter
		}
		val lines: RDD[Array[String]] = rawData.map(_.split("\t"))
		logger.info("共計：" + lines.count.toString() + "筆")
		//----------------------2.建立訓練評估所需資料 RDD[LabeledPoint]-------------
		val categoriesMap: Map[String, Int] = lines.map(fields => fields(3)).distinct.collect.zipWithIndex.toMap
		val labelpointRDD: RDD[LabeledPoint] = lines.map { fields =>
			val trFields: Array[String] = fields.map(_.replaceAll("\"", ""))
			val categoryFeaturesArray: Array[Double] = Array.ofDim[Double](categoriesMap.size)
			val categoryIdx: Int = categoriesMap(fields(3))
			categoryFeaturesArray(categoryIdx) = 1
			val numericalFeatures: Array[Double] = trFields.slice(4, fields.size - 1)
				.map(d => if (d == "?") 0.0 else d.toDouble)
			val label: Int = trFields(fields.size - 1).toInt
			LabeledPoint(label, Vectors.dense(categoryFeaturesArray ++ numericalFeatures))
		}

		val featuresData: RDD[Vector] = labelpointRDD.map(labelpoint => labelpoint.features)
		val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(featuresData)
		val scaledRDD = labelpointRDD.map(labelpoint => LabeledPoint(labelpoint.label, stdScaler.transform(labelpoint.features)))
		//----------------------3.以隨機方式將資料分為3部份並且回傳-------------
		val Array(trainData, validationData, testData) = scaledRDD.randomSplit(Array(0.8, 0.1, 0.1))
		logger.info("將資料分trainData:" + trainData.count() + "   validationData:" + validationData.count() + "   testData:" + testData.count())
		return (trainData, validationData, testData, categoriesMap)
	}

	def PredictData(sc: SparkContext, model: LogisticRegressionModel, categoriesMap: Map[String, Int], testTSVPath: Path, outputDirPath: Path): Unit = {
 
		//----------------------1.匯入轉換資料-------------
		logger.info("開始匯入資料...")
		val rawDataWithHeader = sc.textFile(testTSVPath.toString())
		val rawData = rawDataWithHeader.mapPartitionsWithIndex {
			(idx, iter) => if (idx == 0) iter.drop(1) else iter
		}
		val lines = rawData.map(_.split("\t"))
		logger.info("共計：" + lines.count.toString() + "筆")
		//----------------------2.建立訓練評估所需資料 RDD[LabeledPoint]-------------
		val labelpointRDD = lines.map { fields =>
			val trimmed = fields.map(_.replaceAll("\"", ""))
			val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
			val categoryIdx = categoriesMap(fields(3))
			categoryFeaturesArray(categoryIdx) = 1
			val numericalFeatures = trimmed.slice(4, fields.size)
				.map(d => if (d == "?") 0.0 else d.toDouble)

			val label = 0
			val url = trimmed(0)
			(LabeledPoint(label, Vectors.dense(categoryFeaturesArray ++ numericalFeatures)), url)
		}
		val featuresRDD = labelpointRDD.map { case (labelpoint, url) => labelpoint.features }
		val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(featuresRDD)
		val scaledRDD = labelpointRDD.map { case (labelpoint, url) => (LabeledPoint(labelpoint.label, stdScaler.transform(labelpoint.features)), url) }

		val predicts = scaledRDD.take(10).map {
			case (labelpoint, url) =>
				val predict = model.predict(labelpoint.features)
				var predictDesc = { predict match { case 0 => "暫時性網頁(ephemeral)"; case 1 => "長青網頁(evergreen)"; } }
				val predictOutput = " 網址：  " + url + "==>預測:" + predictDesc
				logger.info(predictOutput)
				(predictOutput)

		}
		//結果儲存到書出資料夾
		val predictRDD = sc.parallelize(predicts);
		predictRDD.saveAsTextFile(outputDirPath.toString())

	}
	def trainEvaluate(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): LogisticRegressionModel = {
		logger.info("開始訓練...")
		val (model, time) = trainModel(trainData, 5, 50, 0.5)
		logger.info("訓練完成,所需時間:" + time + "毫秒")
		val AUC = evaluateModel(model, validationData)
		logger.info("評估結果AUC=" + AUC)
		return (model)
	}

	def trainModel(trainData: RDD[LabeledPoint], numIterations: Int, stepSize: Double, miniBatchFraction: Double): (LogisticRegressionModel, Double) = {
		val startTime = new DateTime()
		val model = LogisticRegressionWithSGD.train(trainData, numIterations, stepSize, miniBatchFraction)
		val endTime = new DateTime()
		val duration = new Duration(startTime, endTime)
		(model, duration.getMillis())
	}

	def evaluateModel(model: LogisticRegressionModel, validationData: RDD[LabeledPoint]): (Double) = {

		val scoreAndLabels = validationData.map { data =>
			var predict = model.predict(data.features)
			(predict, data.label)
		}
		val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
		val AUC = Metrics.areaUnderROC
		(AUC)
	}

	def testModel(model: LogisticRegressionModel, testData: RDD[LabeledPoint]): Unit = {
		val auc = evaluateModel(model, testData)
		logger.info("使用testata測試,結果 AUC:" + auc)
		logger.info("最佳模型使用testData前50筆資料進行預測:")
		val PredictData = testData.take(50)
		PredictData.foreach { data =>
			val predict = model.predict(data.features)
			val result = (if (data.label == predict) "正確" else "錯誤")
			logger.info("實際結果:" + data.label + "預測結果:" + predict + result + data.features)
		}

	}

	def parametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): LogisticRegressionModel = {

		logger.info("-----所有參數交叉評估找出最好的參數組合---------")
		val bestModel = evaluateAllParameter(trainData, validationData, Array(1, 3, 5, 10),
			Array(10, 50, 100), Array(0.5, 0.8, 1))
		return (bestModel)
	}

	def evaluateParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint],
		evaluateParameter: String, numIterationsArray: Array[Int], stepSizeArray: Array[Double], miniBatchFractionArray: Array[Double]): Unit =
		{
			var dataBarChart = new DefaultCategoryDataset()
			var dataLineChart = new DefaultCategoryDataset()
			for (numIterations <- numIterationsArray; stepSize <- stepSizeArray; miniBatchFraction <- miniBatchFractionArray) {
				val (model, time) = trainModel(trainData, numIterations, stepSize, miniBatchFraction)
				val auc = evaluateModel(model, validationData)
				val parameterData =
					evaluateParameter match {
						case "numIterations" => numIterations;
						case "stepSize" => stepSize;
						case "miniBatchFraction" => miniBatchFraction
					}
				dataBarChart.addValue(auc, evaluateParameter, parameterData.toString())
				dataLineChart.addValue(time, "Time", parameterData.toString())

			}

			//      Chart.plotBarLineChart("LogisticRegressionWithSGD evaluations " + evaluateParameter, evaluateParameter, "AUC", 0.48, 0.7, "Time", dataBarChart, dataLineChart)
		}

	def evaluateAllParameter(trainData: RDD[LabeledPoint],
		validationData: RDD[LabeledPoint],
		numIterationsArray: Array[Int],
		stepSizeArray: Array[Double],
		miniBatchFractionArray: Array[Double]): LogisticRegressionModel =
		{
			val evaluations =
				for (numIterations <- numIterationsArray; stepSize <- stepSizeArray; miniBatchFraction <- miniBatchFractionArray) yield {
					val (model, time) = trainModel(trainData, numIterations, stepSize, miniBatchFraction)
					val auc = evaluateModel(model, validationData)
					(numIterations, stepSize, miniBatchFraction, auc)
				}
			val BestEval = (evaluations.sortBy(_._4).reverse)(0)
			logger.info("調校後最佳參數：numIterations:" + BestEval._1 + "  ,stepSize:" + BestEval._2 + "  ,miniBatchFraction:" + BestEval._3
				+ "  ,結果AUC = " + BestEval._4)
			val (bestModel, time) = trainModel(trainData: RDD[LabeledPoint], BestEval._1, BestEval._1, BestEval._3)
			return bestModel
		}

	def SetLogger() = {
		Logger.getLogger("org").setLevel(Level.OFF)
		Logger.getLogger("com").setLevel(Level.OFF)
		System.setProperty("spark.ui.showConsoleProgress", "false")
		Logger.getRootLogger().setLevel(Level.OFF);
	}
	def printHelp(options: Options) = {
		val helpFormatter = new HelpFormatter();
		helpFormatter.printHelp("logreg [options]", options);
	}
}