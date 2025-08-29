import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime
from scipy.stats import f_oneway, chi2_contingency
from statsmodels.formula.api import ols
from ddditai.data.c_data_preparation.a_data_cleaning.data_cleaning import data_cleaning_mlflow_run


# --- OUTLIERS DETECTION FUNCTION (IQR) ---
def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

# --- MAIN MLFLOW PIPELINE ---
EXPERIMENT_NAME = "Sketchfab_Experiment"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# ATTENTION: at the moment mlflow run locally, if moved on a VM or external server make sure to change the path
local_appdata = Path(os.environ['LOCALAPPDATA'])
artifact_base_folder = local_appdata / "MLflow" / "artifacts"
artifact_base_folder.mkdir(parents=True, exist_ok=True)

experiment_description = (
    "This experiment implements commit 7dc8442 of the Data Understanding document and Data Preparation document."
)
experiment_tags = {
    "mlflow.note.content": experiment_description,
}

if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=f"file:///{artifact_base_folder.resolve().as_posix()}",
        tags=experiment_tags
    )

mlflow.set_experiment(EXPERIMENT_NAME)

def analyze_mlflow_run(run_id: str = None, artifact_path: str = None):
    artifact_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    csv_files = [f for f in os.listdir(artifact_local_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV found in artifact folder")
    csv_file_path = os.path.join(artifact_local_path, csv_files[0])
    df = pd.read_csv(csv_file_path)

    # Create run specific folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Analysis_from_{run_id or 'manual'}_{timestamp}"

    run_folder = artifact_base_folder / run_name
    csv_folder = run_folder / "analysis_results"
    plots_folder = run_folder / "plots"
    hist_folder = plots_folder / "histograms"
    box_folder = plots_folder / "boxplots"

    for folder in [csv_folder, hist_folder, box_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    # Descriptive statistic
    desc = df.describe(include="all").transpose()
    desc["median"] = df.median(numeric_only=True)
    desc.to_csv(csv_folder / "descriptive_statistics.csv")
    print("Descriptive statistics saved")

    # Outliers study with box plots visualization
    outlier_report = {}
    for col in df.select_dtypes(include=np.number).columns:
        mask = detect_outliers_iqr(df[col].dropna())
        outlier_report[col] = mask.sum()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col], flierprops={"marker": "o", "markerfacecolor": "red"})
        plt.title(f"Boxplot - {col}")
        plt.tight_layout()
        plt.savefig(box_folder / f"boxplot_{col}.png")
        plt.close()

    pd.DataFrame.from_dict(outlier_report, orient="index", columns=["outlier_count"]).to_csv(
        csv_folder / "outlier_report.csv"
    )
    print("Outlier report saved")

    # Distributions study with histograms visualization
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), bins=20, kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(hist_folder / f"hist_{col}.png")
            plt.close()
    print("Histograms saved")

    # Correlation study of each feature with associated_tag
    if "associated_tag" in df.columns:
        tag_report = []
        for col in df.columns:
            if col == "associated_tag":
                continue
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                contingency = pd.crosstab(df[col].fillna("NA"), df["associated_tag"])
                chi2_stat, p_val, _, _ = chi2_contingency(contingency)
                tag_report.append([col, "Chi2", chi2_stat, p_val])
            elif pd.api.types.is_numeric_dtype(df[col]):
                groups = [df[df["associated_tag"] == tag][col].dropna() for tag in df["associated_tag"].unique()]
                if len(groups) > 1:
                    F_stat, p_val = f_oneway(*groups)
                    tag_report.append([col, "ANOVA", F_stat, p_val])

        pd.DataFrame(tag_report, columns=["feature", "method", "statistic", "p_value"]).to_csv(
            csv_folder / "tag_correlation.csv", index=False
        )
        print("Tag correlation report saved")

    # Correlation study between material_count/texture_count and pbr_type
    if "pbr_type" in df.columns:
        if "texture_count" in df.columns:
            model_tex = ols('texture_count ~ C(pbr_type)', data=df).fit()
            sm.stats.anova_lm(model_tex, typ=2).to_csv(csv_folder / "anova_texture_count_pbr_type.csv")
            print("ANOVA report saved for texture_count ~ pbr_type")

        if "material_count" in df.columns:
            model_mat = ols('material_count ~ C(pbr_type)', data=df).fit()
            sm.stats.anova_lm(model_mat, typ=2).to_csv(csv_folder / "anova_material_count_pbr_type.csv")
            print("ANOVA report saved for material_count ~ pbr_type")

    # Extreme values detection
    extreme_report = {}
    if "face_count" in df.columns:
        extreme_report["face_count_over_200k"] = (df["face_count"] > 200_000).sum()

    pd.DataFrame(list(extreme_report.items()), columns=["metric", "count"]).to_csv(
        csv_folder / "extreme_values.csv", index=False
    )
    print("Extreme values report saved")

    # Missing values analysis
    missing_report = df.isna().sum().to_frame(name="missing_count")
    missing_report["missing_ratio"] = missing_report["missing_count"] / len(df)

    if "pbr_type" in df.columns:
        empty_pbr = df["pbr_type"].isna().sum() + (df["pbr_type"] == "").sum()
        missing_report.loc["pbr_type_empty"] = [empty_pbr, empty_pbr / len(df)]

    missing_report.to_csv(csv_folder / "missing_report.csv")
    print("Missing values report saved")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact(str(csv_folder))
        mlflow.log_artifact(str(box_folder))
        mlflow.log_artifact(str(hist_folder))
        print(f"Run ID: {run.info.run_id} - Analysis artifacts logged successfully.")
        print(f"Analysis completed. Files saved in: {run_folder}")

        if mlflow.active_run():
            mlflow.end_run()

        data_cleaning_mlflow_run(run_id, artifact_path)


if __name__ == "__main__":
    # This main can be used for manual analysis of a specif mlflow run that produced a csv
    analyze_mlflow_run("b0dc176d6f9c414b9f056783f7d69b4f", "csv")