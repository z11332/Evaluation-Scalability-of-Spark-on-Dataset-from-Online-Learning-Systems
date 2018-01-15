
import csv
import pyspark
from pyspark import SparkContext,SparkConf
import datetime
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

starttime = datetime.datetime.now()

conf=pyspark.SparkConf().setAppName("RF_testCategorical2M2").set("spark.executor.memory", "10g").set("spark.executor.instances","2").set("spark.yarn.executor.memoryOverhead", "1000M").set("spark.executor.cores", "2")
sc = pyspark.SparkContext(conf=conf)

spark = SparkSession(sc)


rdd = sc.textFile("gs://dataproc-9b7bf30c-2812-4f5a-b206-49986407c6d2-us-east1/google-cloud-dataproc-metainfo/82ffeb2b-a017-445c-b651-c26f3267d85d/jetcluster-m/dataset/Whole2006_2007Trimed2.csv")

rdd = rdd.mapPartitions(lambda x: csv.reader(x))

#Create the Dataframe of dataset
header = rdd.first()
rdd = rdd.filter(lambda x: x!= header)
df = spark.createDataFrame(rdd,header)

#Seperate variables into Sparse Vectors for KC_Opp and Categorical ones
df.cache()
#One-Hot categorical variables(only select 4, the rest one with too many values)
categoricalColumns = ['ProblemNameNominal','StepNameNominal','ProbleUnitNominal','ProbeSectionNominal']
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Indexed")
    model_s = stringIndexer.fit(df)
    df = model_s.transform(df)
    oneHotEncoder = OneHotEncoder(inputCol=categoricalCol+"Indexed", outputCol=categoricalCol+"classVec")
    df = oneHotEncoder.transform(df)

vecAssembler = VectorAssembler(inputCols=["ProblemNameNominalclassVec",                                          "StepNameNominalclassVec","ProbleUnitNominalclassVec","ProbeSectionNominalclassVec"], outputCol="categAssembVec")
df_New=vecAssembler.transform(df)


df_New=df_New.withColumn("label",df_New["CorrectFirstAttempt"].cast(IntegerType()))

(trainingData, testData) = df_New.randomSplit([0.7, 0.3], seed = 100)

# Create an initial RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="categAssembVec")

# Train model with Training Data
rfModel = rf.fit(trainingData)
predictions = rfModel.transform(testData)

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
#Eval=evaluator.evaluate(predictions)


Eval2=evaluator.evaluate(predictions)

endtime = datetime.datetime.now()
print("Prediction RMSE with 10-folded crosValidation:", Eval2)
print("Running time",(endtime - starttime).seconds,"seconds")

