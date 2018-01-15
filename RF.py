
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

starttime = datetime.datetime.now()
conf=pyspark.SparkConf().setAppName("DT_testCategorical").set("spark.executor.memory", "8g").set("spark.executor.instances","1").set("spark.yarn.executor.memoryOverhead", "1000M").set("spark.executor.cores", "2")
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

rdd = sc.textFile("gs://dataproc-9b7bf30c-2812-4f5a-b206-49986407c6d2-us-east1/google-cloud-dataproc-metainfo/82ffeb2b-a017-445c-b651-c26f3267d85d/jetcluster-m/dataset/WholeDataTrimed1117.csv")
rdd = rdd.mapPartitions(lambda x: csv.reader(x))

#Create the Dataframe of dataset
header = rdd.first()
rdd = rdd.filter(lambda x: x!= header)
df = spark.createDataFrame(rdd,header)
# -------------------Pre-process data------------------
#Seperate variables into Sparse Vectors for KC_Opp and Categorical ones

df.cache()
#One-Hot categorical variables(only select 4, the rest one with too many values)
categoricalColumns = ['ProbleNameNominal','StepNameNominal','Proble_UnitNominal','Porble_SectionNominal']
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Indexed")
    model_s = stringIndexer.fit(df)
    df = model_s.transform(df)
    oneHotEncoder = OneHotEncoder(inputCol=categoricalCol+"Indexed", outputCol=categoricalCol+"classVec")
    df = oneHotEncoder.transform(df)
    
numericCols = ["ZRow_T", "ZHints_T", "ZIncorrects_T", "ZCorrects_T", "ZRow_PN", "ZHints_PN","ZIncorrect_PN","ZCorrects_PN",\
              "ZRow_PU","ZHints_PU","ZIncorrect_PU","ZCorrects_PU",\
              "ZRow_PS","ZHints_PS","ZIncorrect_PS","ZCorrects_PS",\
               "ZKC1_row","ZKC1_hint","ZKC1_incorrect","ZKC1_corrects","ZKC2_row","ZKC2_hint","ZKC2_incorrect","ZKC2_corrects",\
              "ZKC3_row","ZKC3_hint","ZKC3_incorrect","ZKC3_corrects","ZKC4_row","ZKC4_hint","ZKC4_incorrect","ZKC4_corrects",\
              "ZKC5_row","ZKC5_hint","ZKC5_incorrect","ZKC5_corrects",\
              "ZKC_Opp1","ZKC_Opp2","ZKC_Opp3","ZKC_Opp4","ZKC_Opp5","ZKC_Opp6",\
              "ZProblemView","ZKCnumber"]

#Asseble all categorical variables in to one vector feature called "categAssembVec"
vecAssembler = VectorAssembler(inputCols=["ProbleNameNominalclassVec",\
                                          "StepNameNominalclassVec","Proble_UnitNominalclassVec","Porble_SectionNominalclassVec"], outputCol="categAssembVec")
df_New=vecAssembler.transform(df)



# Change numberic columns from String to Double
for numColumn in numericCols:
    df_New = df_New.withColumn(numColumn+"Num", df_New[numColumn].cast(DoubleType()))
    
#Asseble all numberical variables in to one vector feature called "numbAssembVec"
vecAssembler = VectorAssembler(inputCols=["ZRow_TNum", "ZHints_TNum", "ZIncorrects_TNum", "ZCorrects_TNum", "ZRow_PNNum", "ZHints_PNNum","ZIncorrect_PNNum","ZCorrects_PNNum",\
             "ZRow_PUNum","ZHints_PUNum","ZIncorrect_PUNum","ZCorrects_PUNum",\
            "ZRow_PSNum","ZHints_PSNum","ZIncorrect_PSNum","ZCorrects_PSNum",\
              "ZKC1_rowNum","ZKC1_hintNum","ZKC1_incorrectNum","ZKC1_correctsNum","ZKC2_rowNum","ZKC2_hintNum","ZKC2_incorrectNum","ZKC2_correctsNum",\
              "ZKC3_rowNum","ZKC3_hintNum","ZKC3_incorrectNum","ZKC3_correctsNum","ZKC4_rowNum","ZKC4_hintNum","ZKC4_incorrectNum","ZKC4_correctsNum",\
            "ZKC5_rowNum","ZKC5_hintNum","ZKC5_incorrectNum","ZKC5_correctsNum",\
             "ZKC_Opp1Num","ZKC_Opp2Num","ZKC_Opp3Num","ZKC_Opp4Num","ZKC_Opp5Num","ZKC_Opp6Num",\
              "ZProblemViewNum","ZKCnumberNum"], outputCol="numbAssembVec")
df_New=vecAssembler.transform(df_New)

#Asseble all ZKC variables in to one vector feature called "zKCAssembVec"
#vecAssembler = VectorAssembler(inputCols=["ZKC_Opp1Num","ZKC_Opp2Num","ZKC_Opp3Num","ZKC_Opp4Num","ZKC_Opp5Num","ZKC_Opp6Num",\
#              "ZKCnumberNum"], outputCol="ZKCAssembVec")
#df_New=vecAssembler.transform(df_New)

#Assemble all pridictable features in one vector feature called "allAssembVec"
#vecAssembler = VectorAssembler(inputCols=["categAssembVec","numbAssembVec","ZKCAssembVec"], outputCol="allAssembVec")
#df_New=vecAssembler.transform(df_New)


df_New=df_New.withColumn("label",df_New["CorrectFirstAttempt"].cast(IntegerType()))

#---------------dataset split to 70% - 30%---------------------------------------

(trainingData, testData) = df_New.randomSplit([0.7, 0.3], seed = 100)
trainingData.cache()
testData.cache()


#----------Random Forest model build---------------------------

# Create an initial RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="categAssembVec")

# Train model with Training Data
rfModel = rf.fit(trainingData)

#----------------LR model evaluation--------------------------------
#----The categoricalVector pridiction result around 0.72-------------
#-----The ZKCnum vector pridiction result aroudn 0.57----------------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions = rfModel.transform(testData)

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
Eval=evaluator.evaluate(predictions)
print(Eval)


