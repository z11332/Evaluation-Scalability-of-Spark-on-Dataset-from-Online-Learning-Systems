
import csv
import pyspark
from pyspark import SparkContext,SparkConf
import datetime
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

starttime = datetime.datetime.now()
conf=pyspark.SparkConf().setAppName("SVM_testCategorical2M32").set("spark.executor.memory", "10g").set("spark.executor.instances","32").set("spark.yarn.executor.memoryOverhead", "1000M").set("spark.executor.cores", "3")
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)
rdd = sc.textFile("gs://dataproc-9b7bf30c-2812-4f5a-b206-49986407c6d2-us-east1/google-cloud-dataproc-metainfo/82ffeb2b-a017-445c-b651-c26f3267d85d/jetcluster-m/dataset/Whole2006_2007Trimed2.csv")
rdd = rdd.mapPartitions(lambda x: csv.reader(x))

#Create the Dataframe of dataset
header = rdd.first()
rdd = rdd.filter(lambda x: x!= header)
df = spark.createDataFrame(rdd,header)
df.first()

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

#Asseble all categorical variables in to one vector feature called "categAssembVec"
vecAssembler = VectorAssembler(inputCols=["ProblemNameNominalclassVec",                                          "StepNameNominalclassVec","ProbleUnitNominalclassVec","ProbeSectionNominalclassVec"], outputCol="categAssembVec")
df_New=vecAssembler.transform(df)

df_New=df_New.withColumn("label",df_New["CorrectFirstAttempt"].cast(IntegerType()))

(trainingData, testData) = df_New.randomSplit([0.7, 0.3], seed = 100)

trainingData.cache()
testData.cache()


# Create initial SVM Model
lsvc = LinearSVC(labelCol="label", featuresCol="categAssembVec", maxIter=3, regParam=0.1)
# Train model with Training Data
#lsvcModel = lsvc.fit(trainingData)


#predictions = lsvcModel.transform(testData)

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
#evaluator.evaluate(predictions)
# Create ParamGrid for Cross Validation
#paramGrid = ParamGridBuilder().addGrid(lsvc.regParam, [0.1, 0.5, 2.0]).addGrid(lsvc.maxIter, [5, 10, 20]).build()
paramGrid = ParamGridBuilder().addGrid(lsvc.maxIter, [1, 2, 4]).build()
# Create 5-fold CrossValidator
clsvc = CrossValidator(estimator=lsvc, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)

# Run cross validations
clsvcModel = clsvc.fit(trainingData)

# Use test set here so we can measure the accuracy of our model on new data
predictions = clsvcModel.transform(testData)

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
Eval2=evaluator.evaluate(predictions)

endtime = datetime.datetime.now()
print("Prediction RMSE with 10-folded crosValidation:", Eval2)
print("Running time",(endtime - starttime).seconds,"seconds")

