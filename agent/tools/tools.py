import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.gmm import GMM
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.utils.data import generate_data
from pyod.utils.utility import standardizer
from pyod.models.combination import average


# =====================================================
# HELPERS
# =====================================================

def _validate_dataframe(df):

    if df is None:
        raise ValueError(
            "No dataset loaded."
        )

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Expected pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )


def _numeric_columns(df):

    return (
        df.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )


def _categorical_columns(df):

    return (
        df.select_dtypes(
            exclude=[np.number]
        )
        .columns
        .tolist()
    )


# =====================================================
# DATA LOADING
# =====================================================

def load_data(
    file_path: str
):

    df = pd.read_csv(
        file_path
    )

    return {
        "df": df,
        "rows": len(df),
        "columns": df.shape[1],
        "column_names": (
            df.columns.tolist()
        )
    }


# =====================================================
# SCHEMA
# =====================================================

def get_schema(df):

    _validate_dataframe(df)

    return {
        "columns": [
            {
                "column": c,
                "dtype": str(
                    df[c].dtype
                ),
                "null_pct": float(
                    df[c]
                    .isna()
                    .mean()
                )
            }
            for c in df.columns
        ],
        "n_cols": len(df.columns)
    }


# =====================================================
# MISSINGNESS
# =====================================================

def get_missingness_summary(
    df
):

    _validate_dataframe(df)

    return sorted(
        [
            {
                "column": c,
                "missing_pct": float(
                    df[c]
                    .isna()
                    .mean()
                )
            }
            for c in df.columns
        ],
        key=lambda x:
            -x["missing_pct"]
    )


# =====================================================
# NUMERIC SUMMARY
# =====================================================

def get_numeric_summary(df):

    _validate_dataframe(df)

    results = []

    for col in _numeric_columns(df):

        s = df[col].dropna()

        results.append({
            "column": col,
            "mean": (
                float(s.mean())
                if len(s)
                else None
            ),
            "std": (
                float(s.std())
                if len(s) > 1
                else None
            ),
            "min": (
                float(s.min())
                if len(s)
                else None
            ),
            "max": (
                float(s.max())
                if len(s)
                else None
            )
        })

    return results


# =====================================================
# CATEGORICAL SUMMARY
# =====================================================

def get_categorical_summary(
    df
):

    _validate_dataframe(df)

    return [
        {
            "column": c,
            "top_values":
                df[c]
                .value_counts()
                .head(5)
                .to_dict()
        }
        for c in _categorical_columns(df)
    ]


# =====================================================
# PROFILE DATASET
# =====================================================

def profile_dataset(df):

    _validate_dataframe(df)

    return {
        "row_count": len(df),

        "column_count":
            df.shape[1],

        "schema":
            get_schema(df),

        "missing":
            get_missingness_summary(df),

        "numeric":
            get_numeric_summary(df),

        "categorical":
            get_categorical_summary(df)
    }


# =====================================================
# SYNTHETIC DATA
# =====================================================

def DGP(
    contamination=0.05,
    n_train=500,
    n_test=500,
    n_features=6,
    random_state=123,
    shift=True,
    return_df=True
):

    X_train, X_test, y_train, y_test = generate_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        contamination=contamination,
        random_state=random_state)

    X_train = 5 - X_train
    X_test = 5 - X_test

    if return_df:
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

    return X_train, X_test, y_train, y_test


# =====================================================
# MODEL FACTORY
# =====================================================

def _build(
    model_name,
    contamination
):

    models = {
        "hbos": HBOS,    
        "ecod": ECOD,
        "iforest": IForest,
        "pca": PCA,
        "ocsvm": OCSVM,        
        "gmm": GMM,
        "knn": KNN,
        "lof": LOF,
        "cblof": CBLOF,
        "autoencoder": AutoEncoder,
        "vae": VAE,       
    }

    return (
        models.get(
            model_name,
            IForest
        )
        (
            contamination=
            contamination
        )
    )


# =====================================================
# RUN MODEL
# =====================================================

def run_pyod_model(
    df,
    model_name="iforest",
    contamination=0.05
):

    _validate_dataframe(df)

    cols = _numeric_columns(df)

    if len(cols) == 0:

        raise ValueError(
            "No numeric columns found."
        )

    X = df[cols]

    X = SimpleImputer(
        strategy="median"
    ).fit_transform(X)

    X = StandardScaler(
    ).fit_transform(X)

    model = _build(
        model_name,
        contamination
    )

    model.fit(X)

    scored_df = df.copy()

    scored_df[
        "anomaly_label"
    ] = model.labels_

    scored_df[
        "anomaly_score"
    ] = model.decision_scores_

    return {
        "model": model,

        "model_name":
            model_name,

        "X_processed":
            X,

        "threshold":
            getattr(
                model,
                "threshold_",
                None
            ),

        "summary": {
            "count": int(
                scored_df[
                    "anomaly_label"
                ].sum()
            ),

            "rate": float(
                scored_df[
                    "anomaly_label"
                ].mean()
            )
        },

        "scored_df":
            scored_df
    }


# =====================================================
# ANOMALY SUMMARY
# =====================================================

def summarize_anomalies(
    scored_df
):

    if scored_df is None:

        return {
            "message":
                "Run a model first."
        }

    total = len(scored_df)

    anomalies = int(
        scored_df[
            "anomaly_label"
        ].sum()
    )

    return {
        "total": total,
        "anomalies":
            anomalies,
        "rate":
            anomalies / total
    }


# =====================================================
# TOP ANOMALIES
# =====================================================

def get_top_anomalies(
    scored_df,
    n=20
):

    if scored_df is None:
        return []

    return (
        scored_df
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .head(n)
        .to_dict(
            "records"
        )
    )


# =====================================================
# EXPLAIN ANOMALIES
# =====================================================

def explain_anomalies(
    scored_df,
    top_n=10
):

    if scored_df is None:
        return []

    numeric_cols = [

        c

        for c in scored_df
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns

        if c not in [
            "anomaly_label",
            "anomaly_score"
        ]
    ]

    normal_means = (
        scored_df[
            scored_df[
                "anomaly_label"
            ] == 0
        ][numeric_cols]
        .mean()
    )

    anomalies = (
        scored_df[
            scored_df[
                "anomaly_label"
            ] == 1
        ]
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .head(top_n)
    )

    output = []

    for idx, row in anomalies.iterrows():

        diffs = (
            row[numeric_cols]
            - normal_means
        ).abs()

        output.append({
            "row": int(idx),

            "score": float(
                row[
                    "anomaly_score"
                ]
            ),

            "top_features":
                diffs
                .sort_values(
                    ascending=False
                )
                .head(3)
                .index
                .tolist()
        })

    return output


# =====================================================
# COMPARE MODELS
# =====================================================

def compare_models(
    df,
    contamination=0.05
):

    _validate_dataframe(df)

    models = [
        "iforest",
        "ecod",
        "knn",
        "lof",
        "copod",
        "hbos"
    ]

    results = []

    for model_name in models:

        result = run_pyod_model(
            df=df,
            model_name=model_name,
            contamination=contamination
        )

        results.append({
            "model": model_name,
            "anomalies":
                result["summary"]["count"],
            "rate":
                result["summary"]["rate"]
        })

    return results


# =====================================================
# TOOL REGISTRY
# =====================================================

TOOLS = [
    load_data,
    profile_dataset,
    get_schema,
    get_missingness_summary,
    get_numeric_summary,
    get_categorical_summary,
    DGP,
    run_pyod_model,
    summarize_anomalies,
    get_top_anomalies,
    compare_models,
    explain_anomalies,
]