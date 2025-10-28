# ==========================
# ETAPA 1: INIȚIALIZARE SPARK
# ==========================

# Importăm biblioteca SparkSession din pyspark.sql
# SparkSession este punctul de pornire pentru toate aplicațiile Spark.
from pyspark.sql import SparkSession

# Creăm o sesiune Spark cu un nume descriptiv.
# getOrCreate() înseamnă: „creează o nouă sesiune dacă nu există deja”.
spark = SparkSession.builder \
    .appName("ACS2013_LearningAnalytics") \
    .getOrCreate()

# Afișăm versiunea Spark pentru a verifica că totul funcționează.
print("Spark version:", spark.version)

# Ca test, putem crea un mic DataFrame local pentru a ne asigura că Spark funcționează corect.
data_test = [(1, "ok"), (2, "merge")]
columns = ["id", "status"]

# Creăm DataFrame-ul test.
df_test = spark.createDataFrame(data_test, columns)

# Afișăm schema pentru a verifica tipurile de date.
df_test.printSchema()

# Afișăm conținutul DataFrame-ului.
df_test.show()

# ================================
# ETAPA 2: ÎNCĂRCAREA ȘI EXPLORAREA DATELOR
# ================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # pentru agregări și calcule
import os

# Creăm o sesiune Spark (dacă nu există deja)
spark = SparkSession.builder.appName("ACS2013_LearningAnalytics").getOrCreate()

# -----------------------------
# 1️⃣ Setăm calea către fișierul CSV
# -----------------------------
# Presupunem că fișierul este în folderul proiectului, în subdirectorul 'data'
# Exemplu: proiectul tău are structura:
#  proiect/
#   ├── main.py
#   ├── data
#   ├────── ACS2013.csv

csv_path = os.path.join("", "data/ACS2013.csv")

# -----------------------------
# 2️⃣ Citim fișierul CSV în Spark
# -----------------------------
# - header=True  → prima linie are numele coloanelor
# - inferSchema=True → Spark detectează automat tipurile de date
# ⚠️ Prima citire poate dura câteva secunde, fiind un fișier mare.
df = spark.read.csv(csv_path, header=True, inferSchema=True)

# -----------------------------
# 3️⃣ Verificăm schema detectată
# -----------------------------
# df.printSchema()  # afișează tipurile de date ale coloanelor

# -----------------------------
# 4️⃣ Vizualizăm primele 5 rânduri
# -----------------------------
# df.show(5, truncate=False)

# -----------------------------
# 5️⃣ Statistici rapide
# -----------------------------
# număr de înregistrări
print(f"Număr total de înregistrări: {df.count():,}")

# nume coloane
print(f"Număr coloane: {len(df.columns)}")
print("Primele 10 coloane:", df.columns[:10])

# -----------------------------
# 6️⃣ Selectăm câteva coloane-cheie
# -----------------------------
# Pentru analiza veniturilor folosim câteva coloane relevante
cols_cheie = ["AGEP", "EDUC", "SEX", "RACE", "INCOME"]

# Filtrăm doar coloanele care există efectiv în dataset (în unele versiuni pot lipsi)
cols_existente = [c for c in cols_cheie if c in df.columns]

# Afișăm un eșantion din aceste coloane
# df.select(*cols_existente).show(10, truncate=False)

# -----------------------------
# 7️⃣ Numărăm valorile lipsă pe fiecare coloană-cheie
# -----------------------------
null_counts = df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in cols_existente
]).collect()[0].asDict()

print("Număr de valori lipsă / coloană:")
for col, val in null_counts.items():
    print(f" - {col}: {val}")

# -----------------------------
# 8️⃣ Statistici descriptive de bază
# -----------------------------


# Afișăm toate coloanele disponibile în datasetul citit
print("Total coloane:", len(df.columns))

# Calculăm media, min, max și deviația standard pentru vârstă și venit
df.select("AGEP", "PINCP").describe().show()

# -----------------------------
# 9️⃣ Verificăm distribuția veniturilor (opțional)
# -----------------------------
# Grupăm pe intervale de venit pentru o vedere rapidă
if "PINCP" in df.columns:
    df.groupBy().agg(
        F.avg("PINCP").alias("Venit_mediu"),
        F.max("PINCP").alias("Venit_maxim"),
        F.min("PINCP").alias("Venit_minim")
    ).show()

# ================================
# ETAPA 3: CURĂȚARE ȘI TRANSFORMARE
# ================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.types import DoubleType

# 1️⃣ Inițializăm sesiunea Spark (dacă nu există deja)
spark = SparkSession.builder.appName("ACS2013_LearningAnalytics").getOrCreate()

# 2️⃣ Citim fișierul CSV (setul de populație)
csv_path = "data/ACS2013.csv"
df = spark.read.csv(csv_path, header=True, inferSchema=True)

# 3️⃣ Selectăm doar coloanele relevante pentru analiza veniturilor
# AGEP - vârstă
# SCHL - nivel educațional
# SEX - gen
# RAC1P - rasă
# PINCP - venit personal (target)
df = df.select("AGEP", "SCHL", "SEX", "RAC1P", "PINCP")

# 4️⃣ Eliminăm rândurile care au valori lipsă (null) în oricare dintre coloanele de interes
df = df.dropna(subset=["AGEP", "SCHL", "SEX", "RAC1P", "PINCP"])

# 5️⃣ Filtrăm veniturile negative (unele coduri pot fi -1 sau valori invalide)
df = df.filter(df["PINCP"] > 0)

# 6️⃣ Convertim coloanele numerice la tip Double (uneori inferSchema le pune ca IntegerType)
df = df.withColumn("AGEP", F.col("AGEP").cast(DoubleType())) \
       .withColumn("SCHL", F.col("SCHL").cast(DoubleType())) \
       .withColumn("SEX", F.col("SEX").cast(DoubleType())) \
       .withColumn("RAC1P", F.col("RAC1P").cast(DoubleType())) \
       .withColumn("PINCP", F.col("PINCP").cast(DoubleType()))

# 7️⃣ Verificăm rapid schema și câteva rânduri
df.printSchema()
df.show(5)

# 8️⃣ Codificăm variabilele categorice (SEX, RAC1P, SCHL)
# StringIndexer transformă valorile numerice/categorice în indecși 0,1,2,...
# (necesar pentru modelare MLlib)
indexer_sex = StringIndexer(inputCol="SEX", outputCol="SEX_idx")
indexer_race = StringIndexer(inputCol="RAC1P", outputCol="RAC1P_idx")
indexer_edu = StringIndexer(inputCol="SCHL", outputCol="SCHL_idx")

# Aplicăm transformările
df = indexer_sex.fit(df).transform(df)
df = indexer_race.fit(df).transform(df)
df = indexer_edu.fit(df).transform(df)

# 9️⃣ Construim vectorul de trăsături (features) pentru MLlib
# Combinăm AGEP, SEX_idx, RAC1P_idx și SCHL_idx într-o singură coloană "features"
assembler = VectorAssembler(
    inputCols=["AGEP", "SEX_idx", "RAC1P_idx", "SCHL_idx"],
    outputCol="features"
)

df = assembler.transform(df)

# 🔟 Verificăm rezultatul final
df.select("AGEP", "SEX_idx", "RAC1P_idx", "SCHL_idx", "PINCP", "features").show(5, truncate=False)

# 11️⃣ (Opțional) Afișăm statistici rapide pentru veniturile curățate
df.select(F.mean("PINCP").alias("Venit mediu"),
          F.max("PINCP").alias("Venit maxim"),
          F.min("PINCP").alias("Venit minim")).show()

# ================================
# ETAPA 4: MODELARE ȘI EVALUARE MLLIB
# ================================

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# 1️⃣ Inițializăm Spark (dacă nu e deja pornit)
spark = SparkSession.builder.appName("ACS2013_LearningAnalytics").getOrCreate()

# 2️⃣ Presupunem că avem deja DataFrame-ul curățat din Etapa 3: `df`
# Cu coloanele: ["AGEP", "SCHL", "SEX", "RAC1P", "PINCP", "features"]

# 3️⃣ Împărțim datele în seturi de antrenare și test
# - 80% pentru antrenare
# - 20% pentru testare
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train set: {train_data.count():,} rânduri")
print(f"Test set: {test_data.count():,} rânduri")

# 4️⃣ Definim evaluatorul comun pentru toate modelele
# Vom folosi metricile RMSE și R²
evaluator_rmse = RegressionEvaluator(
    labelCol="PINCP", predictionCol="prediction", metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="PINCP", predictionCol="prediction", metricName="r2"
)

# =======================================================
# 🟢 MODEL 1: REGRESIE LINIARĂ
# =======================================================
print("\n=== Model 1: Regresie Liniară ===")

# Inițializăm modelul de regresie liniară
lr = LinearRegression(featuresCol="features", labelCol="PINCP", maxIter=100)

# Antrenăm modelul pe datele de training
lr_model = lr.fit(train_data)

# Aplicăm modelul pe datele de test
lr_predictions = lr_model.transform(test_data)

# Calculăm metricile de performanță
rmse_lr = evaluator_rmse.evaluate(lr_predictions)
r2_lr = evaluator_r2.evaluate(lr_predictions)

print(f"RMSE (Regresie Liniară): {rmse_lr:.2f}")
print(f"R² (Regresie Liniară): {r2_lr:.4f}")

# =======================================================
# 🟠 MODEL 2: ARBORE DE DECIZIE
# =======================================================
print("\n=== Model 2: Arbore de Decizie ===")

# Inițializăm modelul
dt = DecisionTreeRegressor(featuresCol="features", labelCol="PINCP", maxDepth=10)

# Antrenăm modelul
dt_model = dt.fit(train_data)

# Predicții
dt_predictions = dt_model.transform(test_data)

# Evaluare
rmse_dt = evaluator_rmse.evaluate(dt_predictions)
r2_dt = evaluator_r2.evaluate(dt_predictions)

print(f"RMSE (Arbore de Decizie): {rmse_dt:.2f}")
print(f"R² (Arbore de Decizie): {r2_dt:.4f}")

# =======================================================
# 🔵 MODEL 3: PĂDURE ALEATORIE (Random Forest)
# =======================================================
print("\n=== Model 3: Pădure Aleatorie ===")

# Inițializăm modelul de tip ensemble
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="PINCP",
    numTrees=20,       # număr de arbori
    maxDepth=10,       # adâncime maximă
    seed=42
)

# Antrenăm modelul
rf_model = rf.fit(train_data)

# Predicții
rf_predictions = rf_model.transform(test_data)

# Evaluare
rmse_rf = evaluator_rmse.evaluate(rf_predictions)
r2_rf = evaluator_r2.evaluate(rf_predictions)

print(f"RMSE (Random Forest): {rmse_rf:.2f}")
print(f"R² (Random Forest): {r2_rf:.4f}")

# =======================================================
# 🔍 COMPARAȚIE FINALĂ
# =======================================================
print("\n=== Rezumat performanță modele ===")
print(f"{'Model':25s} | {'RMSE':>10s} | {'R²':>10s}")
print("-" * 50)
print(f"{'Regresie Liniară':25s} | {rmse_lr:10.2f} | {r2_lr:10.4f}")
print(f"{'Arbore de Decizie':25s} | {rmse_dt:10.2f} | {r2_dt:10.4f}")
print(f"{'Pădure Aleatorie':25s} | {rmse_rf:10.2f} | {r2_rf:10.4f}")



# ================================
# ETAPA 5: VIZUALIZAREA ȘI SALVAREA GRAFICELOR
# ================================

import os
import matplotlib.pyplot as plt

# 1️⃣ Creăm folderul pentru rezultate (dacă nu există)
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# 2️⃣ Definim datele pentru graficul comparativ
models = ["Regresie Liniară", "Arbore de Decizie", "Pădure Aleatorie"]
rmse_values = [rmse_lr, rmse_dt, rmse_rf]
r2_values = [r2_lr, r2_dt, r2_rf]

# =======================================================
# 🟢 GRAFIC 1: Compararea RMSE între modele
# =======================================================
plt.figure(figsize=(8, 5))
bars = plt.bar(models, rmse_values, color=["#b2182b", "#f4a582", "#2166ac"])
plt.title("Compararea erorii RMSE între modele", fontsize=14)
plt.ylabel("RMSE", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adăugăm valorile numerice deasupra barelor
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.0f}",
             ha='center', va='bottom', fontsize=10, color="black")

# Salvăm imaginea
rmse_img_path = os.path.join(results_dir, "comparatie_rmse.png")
plt.tight_layout()
plt.savefig(rmse_img_path)
plt.close()
print(f"✅ Grafic RMSE salvat la: {rmse_img_path}")

# =======================================================
# 🟣 GRAFIC 2: Compararea R² între modele
# =======================================================
plt.figure(figsize=(8, 5))
bars = plt.bar(models, r2_values, color=["#7fc97f", "#fdc086", "#beaed4"])
plt.title("Compararea scorului R² între modele", fontsize=14)
plt.ylabel("R²", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adăugăm valorile numerice deasupra barelor
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}",
             ha='center', va='bottom', fontsize=10, color="black")

# Salvăm imaginea
r2_img_path = os.path.join(results_dir, "comparatie_r2.png")
plt.tight_layout()
plt.savefig(r2_img_path)
plt.close()
print(f"✅ Grafic R² salvat la: {r2_img_path}")

# =======================================================
# 🔵 GRAFIC 3: Scatter plot predicții vs valori reale (pentru Random Forest)
# =======================================================

# Pentru a face graficul, extragem un eșantion mic din setul de test (altfel sunt prea multe puncte)
sample_df = rf_predictions.select("PINCP", "prediction").sample(fraction=0.001, seed=42)

# Convertim la Pandas pentru matplotlib
sample_pd = sample_df.toPandas()

plt.figure(figsize=(6, 6))
plt.scatter(sample_pd["PINCP"], sample_pd["prediction"], alpha=0.4, s=10, color="#377eb8")
plt.plot([0, sample_pd["PINCP"].max()], [0, sample_pd["PINCP"].max()], color="red", lw=2, label="Perfect match")
plt.title("Predicții vs Valori reale (Random Forest)", fontsize=13)
plt.xlabel("Valori reale PINCP")
plt.ylabel("Predicții PINCP")
plt.legend()
plt.grid(alpha=0.5)

# Salvăm imaginea
scatter_img_path = os.path.join(results_dir, "scatter_pred_vs_real_rf.png")
plt.tight_layout()
plt.savefig(scatter_img_path)
plt.close()
print(f"✅ Grafic scatter salvat la: {scatter_img_path}")

# =======================================================
# 🔍 Raport sumar
# =======================================================
print("\n=== GRAFICE SALVATE ===")
print(f"- {rmse_img_path}")
print(f"- {r2_img_path}")
print(f"- {scatter_img_path}")

