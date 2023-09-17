# Databricks notebook source
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

# COMMAND ----------

# DBTITLE 1,Agrupamiento de Variables Cuantitativas
df3.select(['Total_cajas','Pares','Pago']).describe().show(10)

# COMMAND ----------

df3.show()
df3.printSchema()

# COMMAND ----------

# DBTITLE 1,Segmentacion
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
kmeans = KMeans().setK(5).setSeed(1)

# COMMAND ----------

# DBTITLE 0,Modelado del Aprendizaje Supervisado
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer

# COMMAND ----------

Codigo_indexer = StringIndexer(inputCol= 'Codigo', outputCol= 'CodigoIndex')
Codigo_encoder = OneHotEncoder(inputCol = 'CodigoIndex', outputCol= 'CodigoVec')

Material_indexer = StringIndexer(inputCol= 'Material', outputCol= 'MaterialIndex')
Material_encoder = OneHotEncoder(inputCol = 'MaterialIndex', outputCol= 'MaterialVec')

Color_indexer = StringIndexer(inputCol= 'Color', outputCol= 'ColorIndex')
Color_encoder = OneHotEncoder(inputCol = 'ColorIndex', outputCol= 'ColorVec')

Item_indexer = StringIndexer(inputCol= 'Item', outputCol= 'ItemIndex')
Item_encoder = OneHotEncoder(inputCol = 'ItemIndex', outputCol= 'ItemVec')

Boleta_indexer = StringIndexer(inputCol= 'Boleta', outputCol= 'BoletaIndex')
Boleta_encoder = OneHotEncoder(inputCol = 'BoletaIndex', outputCol= 'BoletaVec')

Marca_indexer = StringIndexer(inputCol= 'Marca', outputCol= 'MarcaIndex')
Marca_encoder = OneHotEncoder(inputCol = 'MarcaIndex', outputCol= 'MarcaVec')

Cliente_indexer = StringIndexer(inputCol= 'Cliente', outputCol= 'ClienteIndex')
Cliente_encoder = StringIndexer(inputCol= 'ClienteIndex', outputCol= 'ClienteVec')

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

train_data, test_data = df3.randomSplit([0.7,0.3])

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['CodigoVec','MaterialVec','ColorVec','ItemVec', 'BoletaVec','MarcaVec','ClienteVec','Pares','Total_cajas','Pago'], outputCol= 'features')
pipeline2 = Pipeline(stages= [Codigo_indexer ,Codigo_encoder ,Material_indexer ,Material_encoder ,Color_indexer ,Color_encoder ,Item_indexer, Item_encoder  Marca_indexer,Boleta_indexer ,Boleta_encoder     Marca_encoder,Cliente_indexer , Cliente_encoder  assembler,  kmeans])

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
