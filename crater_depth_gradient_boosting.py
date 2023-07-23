#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:56:51 2023

@author: liang
"""

#Crater depth prediction using Apache Spark

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler, RobustScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import log


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import rand

import pandas as pd

import sys #for exit()



# Initialize SparkSession
spark = SparkSession.builder.getOrCreate()

# Load data

df = pd.read_csv('LU1319373_Wang & Wu_2021.txt', delim_whitespace=True)

data = spark.createDataFrame(df)

data = data.filter(data['Depth(m)'] != 0)

n = 16  # specify the number of partitions

# Apply the transformation to the columns
data = data.withColumn('ln_Diameter', log(data['Diameter(m)']))
data = data.withColumn('ln_Depth', log(data['Depth(m)']))

# Repartition data
data = data.repartition(n)

# Convert data to features column and label column
vectorAssembler = VectorAssembler(inputCols=["Longitude(degree)", "Latitude(degree)", "ln_Diameter"], 
                                   outputCol="features")

# vectorAssembler = VectorAssembler(inputCols=["ln_Diameter"],
#                                   outputCol="features")

data = vectorAssembler.transform(data)


# Initialize RobustScaler
scaler = RobustScaler(inputCol="features", outputCol="scaledFeatures")
scalerModel = scaler.fit(data)

# Transform the data
scaledData = scalerModel.transform(data)

# Split data
(trainingData, testData) = scaledData.randomSplit([0.7, 0.3],seed=42)

# Initialize model
gbt = GBTRegressor(featuresCol="scaledFeatures", labelCol="ln_Depth")

# Train model
model = gbt.fit(trainingData)

# Make predictions
predictions = model.transform(testData)


# Define the evaluator
evaluator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="ln_Depth", metricName="rmse")
evaluator_r2 = RegressionEvaluator(predictionCol="prediction", labelCol="ln_Depth", metricName="r2")

# Make predictions on the training data and calculate RMSE and R2
train_predictions = model.transform(trainingData)
rmse_train = evaluator_rmse.evaluate(train_predictions)
r2_train = evaluator_r2.evaluate(train_predictions)

# Print RMSE and R2 for the training data
print("Training Data: Root Mean Squared Error = " + str(rmse_train))
print("Training Data: R2 = " + str(r2_train))

# Make predictions on the test data and calculate RMSE and R2
test_predictions = model.transform(testData)
rmse_test = evaluator_rmse.evaluate(test_predictions)
r2_test = evaluator_r2.evaluate(test_predictions)

# Print RMSE and R2 for the test data
print("Test Data: Root Mean Squared Error = " + str(rmse_test))
print("Test Data: R2 = " + str(r2_test))


import matplotlib.pyplot as plt
import numpy as np

# Take a sample of 300 data points from the DataFrame
pandas_df = predictions.sample(False, 0.0003).toPandas()

# Reverse the log transformation
pandas_df['Diameter(m)'] = np.exp(pandas_df['ln_Diameter'])
pandas_df['Depth(m)'] = np.exp(pandas_df['ln_Depth'])
pandas_df['Predicted Depth(m)'] = np.exp(pandas_df['prediction'])

# Create a scatter plot of the actual depths
plt.scatter(pandas_df['Diameter(m)'], pandas_df['Depth(m)'], label='Actual Depth')

# Create a scatter plot of the predicted depths
plt.scatter(pandas_df['Diameter(m)'], pandas_df['Predicted Depth(m)'], label='Predicted Depth')

# Set the labels and title
plt.xlabel('Diameter (m)')
plt.ylabel('Depth (m)')
plt.title('Crater Depth Prediction using Gradient Boosting, R²_validation = ' + str(round(r2_test,2)))

# Add legend to distinguish actual and predicted values
plt.legend()

# Display the plot
plt.show()


