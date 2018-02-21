# Evaluation-Scalability-of-Spark-on-Dataset-from-Online-Learning-Systems
The project is my master thesis.
The datasets are from the KDD competetion 2010, which is to predict the student's performance with their learning logs on the Cognitive Learning systems of US.
At the begining of the research my plan was to increase the accuracy of the prediction with different learning algorithms and feature manipulations. While in the middle of the process both my professor and I gaven up this plan because the former 3 years graduate students in this program falled into almost the same strategies as mine. 
So we switched to the direction of the Evaluation scalability of Apache Spark on the predition with these datasets.

The project includes 4 Data Mining algorithms. For the data pre-processing stage I used SQL and SPSS to make feature generation, cleaning, standardization etc. Then on Spark I used Python with MLlib to implement all the prediction algorithms. 
I build a local Hadoop+Spark cluster with 19 machines in York University for the lab of the research.
Second Spark Cluster I used the Google Cloud with 32 nodes.
Four algorithms are programmed with Python. 
1. Decision Tree
2. Random Forest
3. SVM
4. Linear Logistical Regression

The datasets I tried with 0.8M, 1M, 0.5M and 2M cases. 
