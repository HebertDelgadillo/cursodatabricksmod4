-- Databricks notebook source
DROP TABLE IF EXISTS account;
CREATE TABLE account
USING csv
OPTIONS (path "/FileStore/Caso Berka/account.asc",delimiter ";", header "true");

DROP TABLE IF EXISTS card;
CREATE TABLE card
USING csv
OPTIONS (path "/FileStore/Caso Berka/card.asc",delimiter ";", header "true");

DROP TABLE IF EXISTS client;
CREATE TABLE client
USING csv
OPTIONS (path "/FileStore/Caso Berka/client.asc",delimiter ";", header "true");

DROP TABLE IF EXISTS disp;
CREATE TABLE disp
USING csv
OPTIONS (path "/FileStore/Caso Berka/disp.asc",delimiter ";", header "true");

DROP TABLE IF EXISTS district;
CREATE TABLE district
USING csv
OPTIONS (path "/FileStore/Caso Berka/district.asc",delimiter ";", header "true");

DROP TABLE IF EXISTS loan;
CREATE TABLE loan
USING csv
OPTIONS (path "/FileStore/Caso Berka/loan.asc",delimiter ";", header "true");

DROP TABLE IF EXISTS orden;
CREATE TABLE orden
USING csv
OPTIONS (path "/FileStore/Caso Berka/order.asc",delimiter ";", header "true");

DROP TABLE IF EXISTS trans;
CREATE TABLE trans
USING csv
OPTIONS (path "/FileStore/Caso Berka/trans.asc",delimiter ";", header "true");

-- COMMAND ----------

show tables;

-- COMMAND ----------

select * from district

-- COMMAND ----------

select * from orden order by account_id

-- COMMAND ----------

alter view transcompleta as select t.account_id as Cuenta, count(*) as Nro_movimientos,sum(t.amount) as Total_Dinero_movido,
CASE
    WHEN l.status  = "A" THEN "Excelente candidato"
    WHEN l.status  = "B" THEN "Inconfiable"
    WHEN l.status  = "C" THEN "Confiable"
    WHEN l.status  = "D" THEN "Dudoso"
    else "No se presto"
end Confiabilidad,
CASE
    WHEN a.frequency  = "POPLATEK MESICNE" THEN "Uso mensual"
    WHEN a.frequency  = "POPLATEK TYDNE" THEN "Uso semanal"
    when a.frequency  = "POPLATEK PO OBRATU" THEN "Frecuente"
end Frecuencia, d.a3 as Region,d.A11 as Salario_Promedio,d.a14 as Empresarios_en_miles
from trans t 
inner join account a 
on t.account_id =a.account_id
left join loan l 
on l.account_id = a.account_id
left join district d 
on d.a1 = a.district_id 
group by t.account_id, Confiabilidad, Frecuencia, d.A3,d.a11,d.a14; 

select * from transcompleta
