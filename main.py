# ============================================================
# PROIECT LEARNING ANALYTICS - ACS 2013 (PySpark)
# ============================================================
# Acest script:
#  1) Încarcă setul de date ss13pusa.csv (ACS 2013 Population)
#  2) Curăță și transformă datele
#  3) Construiește vectorul de trăsături (features)
#  4) Antrenează 4 modele de regresie (LR, DT, RF, GBT)
#  5) Antrenează un model SVM (LinearSVC) pentru clasificare binară a venitului
#  6) Evaluează modelele și salvează graficele în folderul results/
# ============================================================

import os                         # pentru lucrul cu directoare și căi de fișiere
import matplotlib.pyplot as plt   # pentru generarea și salvarea graficelor

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import StringIndexer, VectorAssembler

from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
)

from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ============================================================
# 0. FUNCȚIE PRINCIPALĂ
# ============================================================

def main():
    # --------------------------------------------------------
    # 0.1. Inițializare SparkSession
    # --------------------------------------------------------
    # Cream o sesiune Spark — punctul de intrare pentru toate operațiile.
    spark = SparkSession.builder \
        .appName("ACS2013_LearningAnalytics") \
        .getOrCreate()

    print("✅ Spark pornit, versiune:", spark.version)

    # Asigurăm folderul pentru rezultate (grafice etc.)
    results_dir = "results-updated"
    os.makedirs(results_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. ÎNCĂRCAREA DATELOR
    # --------------------------------------------------------
    # Presupunem că fișierul ACS2013.csv este în directorul data/
    data_path = os.path.join("data", "ACS2013.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ Nu am găsit fișierul de date la: {os.path.abspath(data_path)}.\n"
            f"   Asigură-te că ai descărcat ACS2013.csv din Kaggle și l-ai pus în folderul data/."
        )

    print(f"\n📥 Încarc datasetul din: {data_path}")

    # Citim CSV-ul cu header și inferSchema pentru tipuri automate
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # print("\n📌 Schema originală (trunchiată):")
    # df.printSchema()

    # --------------------------------------------------------
    # 2. SELECTAREA COLOANELOR RELEVANTE
    # --------------------------------------------------------
    # Folosim:
    #  - AGEP  = vârstă
    #  - SCHL  = nivel educațional
    #  - SEX   = gen
    #  - RAC1P = rasă
    #  - PINCP = venit personal (target pentru regresie)
    cols_interes = ["AGEP", "SCHL", "SEX", "RAC1P", "PINCP"]
    df = df.select(*cols_interes)

    print("\n📌 Primele 5 rânduri din coloanele de interes:")
    df.show(5)

    # --------------------------------------------------------
    # 3. CURĂȚAREA DATELOR
    # --------------------------------------------------------
    # 3.1 Eliminăm rândurile cu valori lipsă în coloanele importante
    df = df.dropna(subset=["AGEP", "SCHL", "SEX", "RAC1P", "PINCP"])

    # 3.2 Eliminăm veniturile <= 0 (coduri invalide / lipsă)
    df = df.filter(df["PINCP"] > 0)

    # 3.3 Conversia tuturor coloanelor la DoubleType pentru MLlib
    df = df.withColumn("AGEP", F.col("AGEP").cast(DoubleType())) \
           .withColumn("SCHL", F.col("SCHL").cast(DoubleType())) \
           .withColumn("SEX", F.col("SEX").cast(DoubleType())) \
           .withColumn("RAC1P", F.col("RAC1P").cast(DoubleType())) \
           .withColumn("PINCP", F.col("PINCP").cast(DoubleType()))

    print("\n📌 Schema după curățare și conversie la DoubleType:")
    df.printSchema()

    # Statistici descriptive pentru vârstă și venit
    print("\n📊 Statistici descriptive (AGEP, PINCP):")
    df.select("AGEP", "PINCP").describe().show()

    # --------------------------------------------------------
    # 4. CODIFICAREA VARIABILELOR CATEGORICE
    # --------------------------------------------------------
    # Folosim StringIndexer pentru:
    #  - SEX   → SEX_idx
    #  - RAC1P → RAC1P_idx
    #  - SCHL  → SCHL_idx

    indexer_sex = StringIndexer(inputCol="SEX",   outputCol="SEX_idx")
    indexer_race = StringIndexer(inputCol="RAC1P", outputCol="RAC1P_idx")
    indexer_edu = StringIndexer(inputCol="SCHL",  outputCol="SCHL_idx")

    df = indexer_sex.fit(df).transform(df)
    df = indexer_race.fit(df).transform(df)
    df = indexer_edu.fit(df).transform(df)

    print("\n📌 Exemple de codificare categorică (SEX, RAC1P, SCHL):")
    df.select("SEX", "SEX_idx", "RAC1P", "RAC1P_idx", "SCHL", "SCHL_idx") \
      .show(5, truncate=False)

    # --------------------------------------------------------
    # 5. CONSTRUIREA VECTORULUI DE TRĂSĂTURI (features)
    # --------------------------------------------------------
    # Combinăm:
    #  - AGEP
    #  - SEX_idx
    #  - RAC1P_idx
    #  - SCHL_idx
    # într-un singur vector "features".
    assembler = VectorAssembler(
        inputCols=["AGEP", "SEX_idx", "RAC1P_idx", "SCHL_idx"],
        outputCol="features"
    )

    df = assembler.transform(df)

    print("\n📌 Exemple din DataFrame-ul final (features + PINCP):")
    df.select("AGEP", "SEX_idx", "RAC1P_idx", "SCHL_idx", "PINCP", "features") \
      .show(5, truncate=False)


    # --------------------------------------------------------
    # 5.1. CLUSTERIZARE CU K-MEANS
    # --------------------------------------------------------
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator

    print("\n======================================")
    print("🔶 CLUSTERIZARE: K-Means pe features")
    print("======================================")

    k = 4  # număr de clustere

    kmeans = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        seed=42
    )

    kmeans_model = kmeans.fit(df)          # ✔ aici df există
    df_clusters = kmeans_model.transform(df)

    print("\n📌 Primele 10 rânduri cu cluster asignat:")
    df_clusters.select(
        "AGEP", "SCHL", "SEX", "RAC1P", "PINCP", "cluster"
    ).show(10, truncate=False)

    # Evaluare Silhouette
    cluster_evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette"
    )
    silhouette = cluster_evaluator.evaluate(df_clusters)
    print(f"\n📊 Scor Silhouette pentru K={k}: {silhouette:.4f}")

    # Centre clustere
    centers = kmeans_model.clusterCenters()
    print("\n📍 Centre clustere:")
    for i, c in enumerate(centers):
        print(f" Cluster {i}: {c}")

    # Statistici pe cluster
    print("\n📊 Statistici agregate pe clustere:")
    df_clusters.groupBy("cluster").agg(
        F.count("*").alias("nr_persoane"),
        F.avg("AGEP").alias("varsta_medie"),
        F.avg("SCHL").alias("educatie_medie"),
        F.avg("PINCP").alias("venit_mediu")
    ).orderBy("cluster").show()


    # --------------------------------------------------------
    # 6. ÎMPĂRȚIREA ÎN SETURI DE TRAIN ȘI TEST
    # --------------------------------------------------------
    # 80% pentru antrenare, 20% pentru test. seed=42 pentru reproductibilitate.
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    print(f"\n📦 Train set: {train_data.count():,} înregistrări")
    print(f"📦 Test  set: {test_data.count():,} înregistrări")

    # Definim evaluatorii pentru regresie: RMSE și R²
    evaluator_rmse = RegressionEvaluator(
        labelCol="PINCP", predictionCol="prediction", metricName="rmse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="PINCP", predictionCol="prediction", metricName="r2"
    )

    # Dicționare pentru a memora rezultatele modelelor de regresie
    rmse_models = {}
    r2_models = {}

    # --------------------------------------------------------
    # 7. MODEL 1 – REGRESIE LINIARĂ
    # --------------------------------------------------------
    print("\n===============================")
    print("🟢 MODEL 1: REGRESIE LINIARĂ")
    print("===============================")

    lr = LinearRegression(
        featuresCol="features",
        labelCol="PINCP",
        maxIter=100
    )

    lr_model = lr.fit(train_data)                 # antrenăm modelul
    lr_predictions = lr_model.transform(test_data)  # prezicem pe test

    rmse_lr = evaluator_rmse.evaluate(lr_predictions)
    r2_lr = evaluator_r2.evaluate(lr_predictions)

    rmse_models["LR"] = rmse_lr
    r2_models["LR"] = r2_lr

    print(f"RMSE (LR): {rmse_lr:.2f}")
    print(f"R²   (LR): {r2_lr:.4f}")

    # --------------------------------------------------------
    # 8. MODEL 2 – ARBORE DE DECIZIE
    # --------------------------------------------------------
    print("\n===============================")
    print("🟠 MODEL 2: ARBORE DE DECIZIE")
    print("===============================")

    dt = DecisionTreeRegressor(
        featuresCol="features",
        labelCol="PINCP",
        maxDepth=10
    )

    dt_model = dt.fit(train_data)
    dt_predictions = dt_model.transform(test_data)

    rmse_dt = evaluator_rmse.evaluate(dt_predictions)
    r2_dt = evaluator_r2.evaluate(dt_predictions)

    rmse_models["DT"] = rmse_dt
    r2_models["DT"] = r2_dt

    print(f"RMSE (DT): {rmse_dt:.2f}")
    print(f"R²   (DT): {r2_dt:.4f}")

    # --------------------------------------------------------
    # 9. MODEL 3 – PĂDURE ALEATORIE (Random Forest)
    # --------------------------------------------------------
    print("\n======================================")
    print("🔵 MODEL 3: PĂDURE ALEATORIE (RF)")
    print("======================================")

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="PINCP",
        numTrees=20,
        maxDepth=10,
        seed=42
    )

    rf_model = rf.fit(train_data)
    rf_predictions = rf_model.transform(test_data)

    rmse_rf = evaluator_rmse.evaluate(rf_predictions)
    r2_rf = evaluator_r2.evaluate(rf_predictions)

    rmse_models["RF"] = rmse_rf
    r2_models["RF"] = r2_rf

    print(f"RMSE (RF): {rmse_rf:.2f}")
    print(f"R²   (RF): {r2_rf:.4f}")

    # --------------------------------------------------------
    # 10. MODEL 4 – GRADIENT-BOOSTED TREES (GBTRegressor)
    # --------------------------------------------------------
    print("\n========================================")
    print("🟣 MODEL 4: GRADIENT-BOOSTED TREES (GBT)")
    print("========================================")

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="PINCP",
        maxDepth=7,      # adâncime mai mică pentru a reduce overfitting
        maxIter=50,      # număr de iteratii (boosting rounds)
        stepSize=0.1,
        seed=42
    )

    gbt_model = gbt.fit(train_data)
    gbt_predictions = gbt_model.transform(test_data)

    rmse_gbt = evaluator_rmse.evaluate(gbt_predictions)
    r2_gbt = evaluator_r2.evaluate(gbt_predictions)

    rmse_models["GBT"] = rmse_gbt
    r2_models["GBT"] = r2_gbt

    print(f"RMSE (GBT): {rmse_gbt:.2f}")
    print(f"R²   (GBT): {r2_gbt:.4f}")

    # --------------------------------------------------------
    # 11. MODEL 5 – SVM (LinearSVC) PENTRU CLASIFICARE BINARĂ
    # --------------------------------------------------------
    print("\n====================================================")
    print("🧩 MODEL 5: SVM (LinearSVC) – CLASIFICARE VENIT BINAR")
    print("====================================================")

    # 11.1 Calculăm mediana venitului din train (prag pentru venit "ridicat")
    median_income = train_data.approxQuantile("PINCP", [0.5], 0.01)[0]
    print(f"Prag (mediană PINCP): {median_income:.2f}")

    # 11.2 Creăm eticheta binară label_bin: 1 dacă PINCP >= mediană, altfel 0
    train_cls = train_data.withColumn(
        "label_bin",
        (F.col("PINCP") >= F.lit(median_income)).cast("int")
    )
    test_cls = test_data.withColumn(
        "label_bin",
        (F.col("PINCP") >= F.lit(median_income)).cast("int")
    )

    # 11.3 Definim modelul LinearSVC (SVM liniar)
    svm = LinearSVC(
        featuresCol="features",
        labelCol="label_bin",
        maxIter=100,
        regParam=0.01
    )

    svm_model = svm.fit(train_cls)
    svm_predictions = svm_model.transform(test_cls)

    # 11.4 Evaluăm clasificarea cu accuracy, F1, precision și recall
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label_bin", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label_bin", predictionCol="prediction", metricName="f1"
    )
    evaluator_prec = MulticlassClassificationEvaluator(
        labelCol="label_bin", predictionCol="prediction", metricName="precisionByLabel"
    )
    evaluator_rec = MulticlassClassificationEvaluator(
        labelCol="label_bin", predictionCol="prediction", metricName="recallByLabel"
    )

    acc_svm = evaluator_acc.evaluate(svm_predictions)
    f1_svm = evaluator_f1.evaluate(svm_predictions)
    prec_svm = evaluator_prec.evaluate(svm_predictions)
    rec_svm = evaluator_rec.evaluate(svm_predictions)

    print(f"Accuracy (SVM):  {acc_svm:.4f}")
    print(f"F1-score (SVM):  {f1_svm:.4f}")
    print(f"Precision (SVM): {prec_svm:.4f}")
    print(f"Recall (SVM):    {rec_svm:.4f}")



    # --------------------------------------------------------
    # 12. VIZUALIZARE – GRAFICE PENTRU REGRESIE
    # --------------------------------------------------------
    print("\n📊 Generez graficele pentru regresie și SVM...")

    # 12.1 Bar chart pentru RMSE (LR, DT, RF, GBT)
    models_reg = list(rmse_models.keys())   # ["LR", "DT", "RF", "GBT"]
    rmse_vals = [rmse_models[m] for m in models_reg]
    r2_vals = [r2_models[m] for m in models_reg]

    # Grafic RMSE
    plt.figure(figsize=(8, 5))
    plt.bar(models_reg, rmse_vals)
    plt.title("RMSE – comparație modele de regresie")
    plt.ylabel("RMSE")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    rmse_img_path = os.path.join(results_dir, "comparatie_rmse_extins.png")
    plt.savefig(rmse_img_path)
    plt.close()

    # Grafic R²
    plt.figure(figsize=(8, 5))
    plt.bar(models_reg, r2_vals)
    plt.title("R² – comparație modele de regresie")
    plt.ylabel("R²")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    r2_img_path = os.path.join(results_dir, "comparatie_r2_extins.png")
    plt.savefig(r2_img_path)
    plt.close()

    # --------------------------------------------------------
    # 13. VIZUALIZARE – PERFORMANȚĂ SVM (Accuracy și F1)
    # --------------------------------------------------------
    plt.figure(figsize=(6, 5))
    metrics_cls = ["Accuracy", "F1-score"]
    vals_cls = [acc_svm, f1_svm]
    plt.bar(metrics_cls, vals_cls)
    plt.title("Performanța clasificării – SVM (venit ≥ mediană?)")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    svm_img_path = os.path.join(results_dir, "comparatie_svm_cls.png")
    plt.savefig(svm_img_path)
    plt.close()

    # --------------------------------------------------------
    # 14. VIZUALIZARE – SCATTER PENTRU PREDICȚII RF VS VALORI REALE
    # --------------------------------------------------------
    # Folosim un eșantion mic pentru scatter, altfel ar fi enorm.
    sample_rf = rf_predictions.select("PINCP", "prediction").sample(
        fraction=0.001, seed=42
    )
    sample_pd = sample_rf.toPandas()  # convertim la pandas pentru matplotlib

    plt.figure(figsize=(6, 6))
    plt.scatter(sample_pd["PINCP"], sample_pd["prediction"], s=10, alpha=0.4)
    max_val = sample_pd["PINCP"].max()
    plt.plot([0, max_val], [0, max_val], color="red", linewidth=2, label="y = x")
    plt.title("Predicții vs valori reale (Random Forest)")
    plt.xlabel("PINCP real")
    plt.ylabel("PINCP prezis")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    scatter_img_path = os.path.join(results_dir, "scatter_pred_vs_real_rf.png")
    plt.savefig(scatter_img_path)
    plt.close()

    # --------------------------------------------------------
    # 15. REZUMAT FINAL
    # --------------------------------------------------------
    print("\n✅ GATA! Rezumat modele regresie:")
    print(f"{'Model':6s} | {'RMSE':>12s} | {'R²':>8s}")
    print("-" * 32)
    for m in models_reg:
        print(f"{m:6s} | {rmse_models[m]:12.2f} | {r2_models[m]:8.4f}")

    print("\n✅ Performanță SVM (clasificare venit ridicat):")
    print(f"Accuracy:  {acc_svm:.4f}")
    print(f"F1-score:  {f1_svm:.4f}")
    print(f"Precision: {prec_svm:.4f}")
    print(f"Recall:    {rec_svm:.4f}")

    print("\n📂 Grafice salvate în folderul:", os.path.abspath(results_dir))
    print(" -", rmse_img_path)
    print(" -", r2_img_path)
    print(" -", svm_img_path)
    print(" -", scatter_img_path)

    # Oprirea sesiunii Spark
    spark.stop()
    print("\n👋 Spark oprit. Script terminat.")


# ============================================================
# PUNCTUL DE INTRARE
# ============================================================

if __name__ == "__main__":
    main()
