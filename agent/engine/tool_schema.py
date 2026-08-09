# engine/tool_schema.py

from google.genai import types


TOOL_SCHEMAS = [
    types.Tool(
        function_declarations=[

            # ============================================
            # DATA LOADING
            # ============================================

            types.FunctionDeclaration(
                name="load_data",
                description=(
                    "Load a CSV dataset from a file path."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "file_path": {
                            "type": "STRING",
                            "description": (
                                "Path to the CSV file"
                            )
                        }
                    },
                    "required": [
                        "file_path"
                    ]
                },
            ),

            # ============================================
            # SYNTHETIC DATA
            # ============================================

            types.FunctionDeclaration(
                name="DGP",
                description=(
                    "Generate a synthetic anomaly detection dataset. "
                    "Use this whenever the user asks for synthetic data, "
                    "sample data, demo data, benchmark data, mock data, "
                    "toy data, random data, generated data, or anomaly "
                    "detection training data."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {

                        "n_train": {
                            "type": "INTEGER",
                            "description": (
                                "Number of training samples"
                            )
                        },

                        "n_test": {
                            "type": "INTEGER",
                            "description": (
                                "Number of testing samples"
                            )
                        },

                        "n_features": {
                            "type": "INTEGER",
                            "description": (
                                "Number of numerical features"
                            )
                        },

                        "contamination": {
                            "type": "NUMBER",
                            "description": (
                                "Fraction of anomalies"
                            )
                        }

                    }
                },
            ),

            # ============================================
            # PROFILING
            # ============================================

            types.FunctionDeclaration(
                name="profile_dataset",
                description=(
                    "Generate a complete dataset profile."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {}
                },
            ),

            types.FunctionDeclaration(
                name="get_schema",
                description=(
                    "Return dataset schema and column types."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {}
                },
            ),

            types.FunctionDeclaration(
                name="get_missingness_summary",
                description=(
                    "Return missing value statistics."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {}
                },
            ),

            types.FunctionDeclaration(
                name="get_numeric_summary",
                description=(
                    "Return summary statistics for numeric columns."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {}
                },
            ),

            types.FunctionDeclaration(
                name="get_categorical_summary",
                description=(
                    "Return summary statistics for categorical columns."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {}
                },
            ),

            # ============================================
            # MODELING
            # ============================================

            types.FunctionDeclaration(
                name="run_pyod_model",
                description=(
                    "Run an anomaly detection model using PyOD."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {

                        "model_name": {
                            "type": "STRING",
                            "enum": [
                                "hbos",    
                                "ecod",
                                "iforest",
                                "pca",
                                "ocsvm",        
                                "gmm",
                                "knn",
                                "lof",
                                "cblof",
                                "autoencoder",
                                "vae",
                            ],
                            "description": (
                                "Anomaly detection model"
                            )
                        },

                        "contamination": {
                            "type": "NUMBER",
                            "description": (
                                "Expected anomaly rate"
                            )
                        }

                    },
                    "required": [
                        "model_name"
                    ]
                },
            ),

            # ============================================
            # RESULTS
            # ============================================

            types.FunctionDeclaration(
                name="summarize_anomalies",
                description=(
                    "Summarize anomaly detection results."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {}
                },
            ),

            types.FunctionDeclaration(
                name="get_top_anomalies",
                description=(
                    "Return the highest-scoring anomalies."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "n": {
                            "type": "INTEGER",
                            "description": (
                                "Number of anomalies to return"
                            )
                        }
                    }
                },
            ),

            types.FunctionDeclaration(
                name="explain_anomalies",
                description=(
                    "Explain the most important anomaly drivers."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "top_n": {
                            "type": "INTEGER",
                            "description": (
                                "Number of anomalies to explain"
                            )
                        }
                    }
                },
            ),

            # ============================================
            # MODEL COMPARISON
            # ============================================

            types.FunctionDeclaration(
                name="compare_models",
                description=(
                    "Compare anomaly detection models and "
                    "return anomaly counts and rates."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "contamination": {
                            "type": "NUMBER",
                            "description": (
                                "Expected anomaly rate"
                            )
                        }
                    }
                },
            ),

        ]
    )
]