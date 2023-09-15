# Databricks notebook source
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS account;
# MAGIC CREATE TABLE account
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/Caso Berka/account.asc",delimiter ";", header "true");
# MAGIC
# MAGIC DROP TABLE IF EXISTS card;
# MAGIC CREATE TABLE card
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/Caso Berka/card.asc",delimiter ";", header "true");
# MAGIC
# MAGIC DROP TABLE IF EXISTS client;
# MAGIC CREATE TABLE client
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/Caso Berka/client.asc",delimiter ";", header "true");
# MAGIC
# MAGIC DROP TABLE IF EXISTS disp;
# MAGIC CREATE TABLE disp
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/Caso Berka/disp.asc",delimiter ";", header "true");
# MAGIC
# MAGIC DROP TABLE IF EXISTS district;
# MAGIC CREATE TABLE district
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/Caso Berka/district.asc",delimiter ";", header "true");
# MAGIC
# MAGIC DROP TABLE IF EXISTS loan;
# MAGIC CREATE TABLE loan
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/Caso Berka/loan.asc",delimiter ";", header "true");
# MAGIC
# MAGIC DROP TABLE IF EXISTS orden;
# MAGIC CREATE TABLE orden
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/Caso Berka/order.asc",delimiter ";", header "true");
# MAGIC
# MAGIC DROP TABLE IF EXISTS trans;
# MAGIC CREATE TABLE trans
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/Caso Berka/trans.asc",delimiter ";", header "true");

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from district

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from orden order by account_id

# COMMAND ----------

# MAGIC %sql
# MAGIC alter view transcompleta as select t.account_id as Cuenta, count(*) as Nro_movimientos,sum(t.amount) as Total_Dinero_movido,
# MAGIC CASE
# MAGIC     WHEN l.status  = "A" THEN "Excelente candidato"
# MAGIC     WHEN l.status  = "B" THEN "Inconfiable"
# MAGIC     WHEN l.status  = "C" THEN "Confiable"
# MAGIC     WHEN l.status  = "D" THEN "Dudoso"
# MAGIC     else "No se presto"
# MAGIC end Confiabilidad,
# MAGIC CASE
# MAGIC     WHEN a.frequency  = "POPLATEK MESICNE" THEN "Uso mensual"
# MAGIC     WHEN a.frequency  = "POPLATEK TYDNE" THEN "Uso semanal"
# MAGIC     when a.frequency  = "POPLATEK PO OBRATU" THEN "Frecuente"
# MAGIC end Frecuencia, d.a3 as Region,d.A11 as Salario_Promedio,d.a14 as Empresarios_en_miles
# MAGIC from trans t 
# MAGIC inner join account a 
# MAGIC on t.account_id =a.account_id
# MAGIC left join loan l 
# MAGIC on l.account_id = a.account_id
# MAGIC left join district d 
# MAGIC on d.a1 = a.district_id 
# MAGIC group by t.account_id, Confiabilidad, Frecuencia, d.A3,d.a11,d.a14; 
# MAGIC
# MAGIC select * from transcompleta

# COMMAND ----------

#import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, DoubleType, FloatType

# Create SparkSession 
spark = SparkSession.builder \
      .master("local[1]") \
            .appName("SparkByExamples.com") \
                  .getOrCreate()
df=spark.sql("select * from transcompleta")

# COMMAND ----------

df.show(10)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df2= spark.sql("select STRING(Cuenta),Double(Nro_Movimientos),Double(Total_Dinero_movido), String(Confiabilidad),String(Frecuencia), String(Region), Double(Salario_Promedio),Double(Empresarios_en_miles) from transcompleta")

# COMMAND ----------

df2.show(10)

# COMMAND ----------

df2.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

df2.sort(F.desc('Total_Dinero_movido')).show(10)

# COMMAND ----------

df2.count()

# COMMAND ----------

df3 = df2.na.drop()

# COMMAND ----------

df2.count()

# COMMAND ----------

# DBTITLE 1,Agrupamiento de Variables Cualitativas
df3.groupBy(F.col('Cuenta')).count().show(5)
df3.groupBy(F.col('Confiabilidad')).count().show(5)
df3.groupBy(F.col('Frecuencia')).count().show(5)
df3.groupBy(F.col('Region')).count().show(5)

# COMMAND ----------

# DBTITLE 1,Agrupamiento de Variables Cuantitativas
df3.select(['Nro_Movimientos','Total_Dinero_movido','Salario_Promedio','Empresarios_en_miles']).describe().show(10)

# COMMAND ----------

df4 = df3.select(['Nro_Movimientos','Total_Dinero_movido','Confiabilidad','Frecuencia','Region','Salario_Promedio','Empresarios_en_miles'])
df4.show(5)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer

# COMMAND ----------

Region_indexer = StringIndexer(inputCol= 'Region', outputCol= 'RegionIndex')
Region_encoder = OneHotEncoder(inputCol = 'RegionIndex', outputCol= 'RegionVec')

Frecuencia_indexer = StringIndexer(inputCol= 'Frecuencia', outputCol= 'FrecuenciaIndex')
Frecuencia_encoder = OneHotEncoder(inputCol = 'FrecuenciaIndex', outputCol= 'FrecuenciaVec')

Confiabilidad_indexer = StringIndexer(inputCol= 'Confiabilidad', outputCol= 'ConfiabilidadIndex')

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['FrecuenciaVec','RegionVec','Nro_Movimientos','Total_Dinero_movido','Salario_Promedio','Empresarios_en_miles'], outputCol= 'features')

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# COMMAND ----------

log_reg_berka = LogisticRegression(featuresCol= 'features', labelCol='ConfiabilidadIndex')

# COMMAND ----------

pipeline = Pipeline(stages= [
    Region_indexer,
    Region_encoder,
    Frecuencia_indexer,
    Frecuencia_encoder,
    Confiabilidad_indexer,
    assembler, 
    log_reg_berka])

# COMMAND ----------

train_data, test_data = df4.randomSplit([0.7,0.3])

# COMMAND ----------

fit_model = pipeline.fit(train_data)

# COMMAND ----------

results = fit_model.transform(test_data)
results.show(5)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

me_eval = BinaryClassificationEvaluator(rawPredictionCol= 'prediction', labelCol = 'ConfiabilidadIndex')

# COMMAND ----------

results.select('ConfiabilidadIndex', 'prediction').show(10)

# COMMAND ----------

# DBTITLE 1,Evaluacion del Modelo
auc = me_eval.evaluate(results)
print("AUC:",auc)
