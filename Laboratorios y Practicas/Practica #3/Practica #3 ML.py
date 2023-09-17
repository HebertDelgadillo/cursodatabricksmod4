# Databricks notebook source
# DBTITLE 1,Preprocesamiento de Datos
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:,.2F}'.format

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS Nova;
# MAGIC CREATE TABLE Nova
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/Practica 3/Datos.csv",delimiter ",", header "true");

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, DoubleType, FloatType

# Create SparkSession 
spark = SparkSession.builder \
      .master("local[1]") \
            .appName("SparkByExamples.com") \
                  .getOrCreate()
df=spark.sql("SELECT * FROM Nova;")

# COMMAND ----------

df.show(5)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df=df.withColumnRenamed("T.cajas", "Total_cajas").withColumnRenamed("Total($)", "Total").withColumnRenamed("Desc($)", "Descuento").withColumnRenamed("Pago($)", "Pago")
df.printSchema()

# COMMAND ----------

df2= df.withColumn("Total_cajas", df["Total_cajas"].cast('float')).withColumn("Pares", df["Pares"].cast('float')).withColumn("Total", df["Total"].cast('float')).withColumn("Descuento", df["Descuento"].cast('float')).withColumn("Pago", df["Pago"].cast('float'))

# COMMAND ----------

df2.printSchema()

# COMMAND ----------

df3=df2.drop('FECHA DE INGRESO')
df3=df3.drop('Factura')
df3=df3.drop('Almacen')
df3=df3.drop('Fecha')
df3=df3.drop('Hora')
df3=df3.drop('Total')
df3=df3.drop('Descuento')
df3=df3.drop('Vendedor')


# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

df3.sort(F.desc('Total')).show(10)

# COMMAND ----------

df3.count()

# COMMAND ----------

df3 = df3.na.drop()

# COMMAND ----------

df3.count()

# COMMAND ----------

# DBTITLE 1,Agrupamiento de Variables Cualitativas
df3.groupBy(F.col('Codigo')).count().show(5)
df3.groupBy(F.col('Material')).count().show(5)
df3.groupBy(F.col('Color')).count().show(5)
df3.groupBy(F.col('Almacen')).count().show(5)
df3.groupBy(F.col('Boleta')).count().show(5)
df3.groupBy(F.col('Marca')).count().show(5)
df3.groupBy(F.col('Cliente')).count().show(5)
df3.groupBy(F.col('Vendedor')).count().show(5)

# COMMAND ----------

# DBTITLE 1,Agrupamiento de Variables Cuantitativas
df3.select(['Total_cajas','Pares','Total','Pago']).describe().show(10)

# COMMAND ----------

# DBTITLE 1,Modelado del Aprendizaje Supervisado (Logistic Regression)
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer

# COMMAND ----------

Codigo_indexer = StringIndexer(inputCol= 'Codigo', outputCol= 'CodigoIndex')
Codigo_encoder = OneHotEncoder(inputCol = 'CodigoIndex', outputCol= 'CodigoVec')

Material_indexer = StringIndexer(inputCol= 'Material', outputCol= 'MaterialIndex')
Material_encoder = OneHotEncoder(inputCol = 'MaterialIndex', outputCol= 'MaterialVec')

Boleta_indexer = StringIndexer(inputCol= 'Boleta', outputCol= 'BoletaIndex')
Boleta_encoder = OneHotEncoder(inputCol = 'BoletaIndex', outputCol= 'BoletaVec')

Marca_indexer = StringIndexer(inputCol= 'Marca', outputCol= 'MarcaIndex')
Marca_encoder = OneHotEncoder(inputCol = 'MarcaIndex', outputCol= 'MarcaVec')

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['CodigoVec','MaterialVec','BoletaVec','MarcaVec','Pares','Total_cajas'], outputCol= 'features')

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

# COMMAND ----------

rf = RandomForestRegressor(labelCol="Pago", featuresCol="features")

# COMMAND ----------

pipeline = Pipeline(stages= [Codigo_indexer, Codigo_encoder,    Material_indexer,    Material_encoder,  Boleta_indexer,    Boleta_encoder,    Marca_indexer,    Marca_encoder,    assembler,    rf])

# COMMAND ----------

(trainingData, testData) = df3.randomSplit([0.8, 0.2])

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 3)]) \
    .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
    .build()
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(labelCol="Pago", predictionCol="prediction", metricName="rmse"),
                          numFolds=3)

# COMMAND ----------

model = pipeline.fit(trainingData)

# COMMAND ----------

predictions = model.transform(testData)
rmse = evaluator.evaluate(predictions)

# COMMAND ----------

predictions.select('Pago', 'prediction').show(10)

# COMMAND ----------

# DBTITLE 1,Evaluacion del Modelo
auc = me_eval.evaluate(results)
print("AUC:",auc)

# COMMAND ----------

# DBTITLE 1,Modelado del Aprendizaje Supervisado (Random Forest)
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=10, maxDepth=6, labelCol="ConfiabilidadIndex", seed=42, leafCol="leafId")
pipelinerf = Pipeline(stages= [
    Region_indexer,
    Region_encoder,
    Frecuencia_indexer,
    Frecuencia_encoder,
    Confiabilidad_indexer,
    assembler, 
    rf])

# COMMAND ----------

fit_modelrf = pipelinerf.fit(train_data)

# COMMAND ----------

resultsrf = fit_modelrf.transform(test_data)
resultsrf.show(5)

# COMMAND ----------

# DBTITLE 1,Evaluacion del Modelo (Random Forest)
auc = me_eval.evaluate(resultsrf)
print("AUC:",auc)

# COMMAND ----------

# DBTITLE 1,Segmentacion
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# COMMAND ----------

dfclus= spark.sql("select Double(Cuenta),Double(Nro_Movimientos),Double(Total_Dinero_movido), String(Confiabilidad),String(Frecuencia), String(Region), Double(Salario_Promedio),Double(Empresarios_en_miles),Double(Habitantes),Double(Urbanizacion),Double(Desempleo),Double(Crimenes) from transcompleta WHERE Confiabilidad = 'No se presto';")
kmeans = KMeans().setK(5).setSeed(1)

# COMMAND ----------

train_data, test_data = dfclus.randomSplit([0.7,0.3])

# COMMAND ----------

pipeline2 = Pipeline(stages= [
    Region_indexer,
    Region_encoder,
    Frecuencia_indexer,
    Frecuencia_encoder,
    assembler, 
    kmeans])

# COMMAND ----------

fit_model2 = pipeline2.fit(train_data)

# COMMAND ----------

predictions2 = fit_model2.transform(test_data)

# COMMAND ----------

predictions2.show(10)

# COMMAND ----------

# DBTITLE 1,Evaluacion del Modelo (KMeans)
evaluador = ClusteringEvaluator()

# COMMAND ----------

silhouette = evaluador.evaluate(predictions2)
print("El coeficiente Silhouette usando distancias euclidianas al cuadrado es = " + str(silhouette))
