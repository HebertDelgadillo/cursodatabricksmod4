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
