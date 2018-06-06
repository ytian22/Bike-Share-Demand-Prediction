from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer    
from pyspark.sql.functions import when
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder

app_name = "sparkml"
conf = SparkConf().setMaster("local").setAppName(app_name)
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)


# read csv file as pyspark framework and convert into spark sql data frame

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true',inferschema='true').load('bikeDF2.csv')

    
targetDf = df.withColumn("precipitation_inches",when(df["precipitation_inches"] == 'T',0).otherwise(df["precipitation_inches"]))

targetDf = targetDf.withColumn("precipitation_inches1", targetDf.precipitation_inches.cast(DoubleType()))

targetDf = targetDf.drop('precipitation_inches')

df = targetDf.drop('unnamed', 'date', 'installation_date', 'city')
                               
df = df.na.drop()   

#converting strings to numeric values


def indexStringColumns(df, cols):
    #variable newdf will be updated several times
    newdf = df
    
    for c in cols:
        #For each given colum, fits StringIndexerModel.
        si = StringIndexer(inputCol=c, outputCol=c+"-num")
        sm = si.fit(newdf)
        #Creates a DataFame by putting the transformed values in the new colum with suffix "-num" 
        #and then drops the original columns.
        #and drop the "-num" suffix. 
        newdf = sm.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-num", c)
    return newdf

dfnumeric = indexStringColumns(df,['start_station_name', 'events', 'dock_count'])

# Merging the data with Vector Assembler.
input_cols = ['zip_code','max_temperature_f','mean_temperature_f','min_temperature_f',
        'max_dew_point_f','mean_dew_point_f','min_dew_point_f','max_humidity','mean_humidity',
        'min_humidity','max_sea_level_pressure_inches', 'mean_sea_level_pressure_inches',
        'min_sea_level_pressure_inches', 'max_visibility_miles','mean_visibility_miles',
        'min_visibility_miles', 'max_wind_Speed_mph', 'mean_wind_speed_mph',
        'max_gust_speed_mph','precipitation_inches1','cloud_cover','wind_dir_degrees',
        'station_id', 'lat','long','holidays','day_of_week','start_station_name',
        'events','dock_count']

#VectorAssembler takes a number of collumn names(inputCols) and output column name (outputCol)
#and transforms a DataFrame to assemble the values in inputCols into one single vector with outputCol.
va = VectorAssembler(outputCol="features", inputCols=input_cols)
#lpoints - labeled data.
df_final = va.transform(dfnumeric).select("features", "label")

#split to training and test, cache the df 
dfSets = df_final.randomSplit([0.8, 0.2], 1)
dfTrain = dfSets[0].cache()
dfTest = dfSets[1].cache()

#fit the model 
rf = RandomForestRegressor(maxDepth=20, maxBins=70)
rfmodel = rf.fit(dfTrain)

#calculate the rmse and prediction
rfpredicts = rfmodel.transform(dfTest)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(rfpredicts)

#cross validation
#cv = CrossValidator().setEstimator(rf).setEvaluator(evaluator).setNumFolds(3)

#paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [30,40,50])\
    #.addGrid(rf.maxDepth, [10,20,30])\
    #.addGrid(rf.maxBins, [70,80,90]).build()
#setEstimatorParamMaps() takes ParamGridBuilder().
#cv.setEstimatorParamMaps(paramGrid)
#cvmodel = cv.fit(dfTrain)

#rmse_best = evaluator.evaluate(cvmodel.bestModel.transform(dfTest))
#print the desired output
print " "
print targetDf.printSchema()
print " "
print("RMSE_baseline = %d" % rmse)
#print " "
#print("RMSE_best = %d" % rmse_best)



sc.stop()


