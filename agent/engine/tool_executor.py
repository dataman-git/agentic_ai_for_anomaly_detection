from tools.tools import *


def execute_tool(
    name,
    args,
    state
):

    df = state.get("df")

    # ==========================================
    # DATA LOADING
    # ==========================================

    if name == "load_data":

        result = load_data(**args)

        state["df"] = result["df"]

        return {
            "rows": result["rows"],
            "columns": result["columns"],
            "column_names": result["column_names"]
        }

    # ==========================================
    # SYNTHETIC DATA
    # ==========================================

    if name == "DGP":
    
        X_train, X_test, y_train, y_test = DGP(**args)
    
        state["df"] = X_train
        state["X_test"] = X_test
        state["y_train"] = y_train
        state["y_test"] = y_test
    
        return {
            "rows": X_train.shape[0],
            "features": X_train.shape[1],
            "anomalies": int(y_train.sum()),
            "contamination": args.get("contamination", 0.05)
        }

    # ==========================================
    # DATASET REQUIRED
    # ==========================================

    if df is None and name not in [
        "load_data",
        "DGP"
    ]:
        return (
            "No dataset loaded. "
            "Load or generate data first."
        )

    # ==========================================
    # PROFILING
    # ==========================================

    if name == "profile_dataset":
        return profile_dataset(df)

    if name == "get_schema":
        return get_schema(df)

    if name == "get_missingness_summary":
        return get_missingness_summary(df)

    if name == "get_numeric_summary":
        return get_numeric_summary(df)

    if name == "get_categorical_summary":
        return get_categorical_summary(df)

    # ==========================================
    # MODELING
    # ==========================================

    if name == "run_pyod_model":

        result = run_pyod_model(
            df=df,
            **args
        )

        state["model"] = result["model"]
        state["model_name"] = result["model_name"]
        state["threshold"] = result["threshold"]
        state["X_processed"] = result["X_processed"]
        state["scored_df"] = result["scored_df"]

        return result["summary"]

    # ==========================================
    # ANOMALY RESULTS
    # ==========================================

    if name == "summarize_anomalies":

        return summarize_anomalies(
            state.get("scored_df")
        )

    if name == "get_top_anomalies":

        return get_top_anomalies(
            state.get("scored_df"),
            n=args.get("n", 20)
        )

    if name == "explain_anomalies":

        return explain_anomalies(
            state.get("scored_df"),
            top_n=args.get("top_n", 10)
        )

    # ==========================================
    # MODEL COMPARISON
    # ==========================================

    if name == "compare_models":

        return compare_models(
            df,
            contamination=args.get(
                "contamination",
                0.05
            )
        )

    # ==========================================
    # UNKNOWN TOOL
    # ==========================================

    return f"Unknown tool: {name}"