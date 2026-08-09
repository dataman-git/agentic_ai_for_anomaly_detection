import os
import sys
import traceback

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT_DIR)

from google.genai import types
from engine.tool_schema import TOOL_SCHEMAS
from engine.tool_executor import execute_tool


MAX_STEPS = 5


def run_agent(
    prompt,
    client,
    state,
    system_prompt
):
    """
    Main agent loop.

    Supports:
    - Multi-step reasoning
    - Tool calling
    - Tool result feedback
    - Conversation state via Streamlit session_state
    """

    full_prompt = (
        system_prompt
        + "\n\nUser: "
        + prompt
    )

    for step in range(MAX_STEPS):

        # =============================================
        # CALL GEMINI
        # =============================================

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    tools=TOOL_SCHEMAS
                ),
            )

        except Exception as e:

            error_text = str(e)

            if "RESOURCE_EXHAUSTED" in error_text:

                return (
                    "Gemini API quota exceeded.\n\n"
                    "You have reached the current rate limit "
                    "for the configured Gemini project.\n"
                    "Please try again later."
                )

            return (
                "Gemini request failed:\n\n"
                f"{error_text}"
            )

        # =============================================
        # SAFETY CHECKS
        # =============================================

        if response is None:

            return "No response returned by Gemini."

        if (
            not hasattr(response, "candidates")
            or not response.candidates
        ):
            return "Gemini returned no candidates."

        candidate = response.candidates[0]

        if (
            not hasattr(candidate, "content")
            or candidate.content is None
            or not candidate.content.parts
        ):
            return (
                "Gemini returned no valid content."
            )

        # =============================================
        # PROCESS PARTS
        # =============================================

        for part in candidate.content.parts:

            # =========================================
            # TOOL CALL
            # =========================================

            if (
                hasattr(part, "function_call")
                and part.function_call is not None
            ):

                try:

                    tool_name = (
                        part.function_call.name
                    )

                    tool_args = (
                        dict(part.function_call.args)
                        if part.function_call.args
                        else {}
                    )

                except Exception as e:

                    return (
                        "Failed to parse tool call:\n\n"
                        f"{str(e)}"
                    )

                print(
                    f"[STEP {step + 1}] "
                    f"Tool={tool_name}"
                )

                print(
                    f"[STEP {step + 1}] "
                    f"Args={tool_args}"
                )

                # =====================================
                # ANOMALY MODEL SAFEGUARD
                # =====================================

                if tool_name == "run_pyod_model":

                    user_prompt_lower = (
                        prompt.lower()
                    )

                    anomaly_terms = [
                        "anomaly",
                        "anomalies",
                        "outlier",
                        "outliers",
                        "detect",
                        "detection",
                        "model",                    
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
                    ]

                    if not any(
                        term in user_prompt_lower
                        for term in anomaly_terms
                    ):

                        return (
                            "Your dataset is loaded and ready.\n\n"
                            "You can ask me to:\n"
                            "- Profile the dataset\n"
                            "- Show schema\n"
                            "- Analyze missing values\n"
                            "- Review numeric statistics\n"
                            "- Review categorical statistics\n"
                            "- Run anomaly detection\n"
                            "- Compare anomaly detection models"
                        )

                # =====================================
                # EXECUTE TOOL
                # =====================================

                try:

                    result = execute_tool(
                        tool_name,
                        tool_args,
                        state,
                    )

                except Exception as e:

                    result = {
                        "tool": tool_name,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }

                result_text = str(result)

                # =====================================
                # TRUNCATE LARGE OUTPUT
                # =====================================

                if len(result_text) > 5000:

                    result_text = (
                        result_text[:5000]
                        + "\n...[TRUNCATED]..."
                    )

                print(
                    f"[STEP {step + 1}] "
                    f"Result={result_text[:500]}"
                )

                # =====================================
                # FEED BACK INTO PROMPT
                # =====================================

                full_prompt += (
                    "\n\n"
                    f"Tool Called: {tool_name}\n"
                    f"Tool Output:\n{result_text}\n"
                )

                # Continue reasoning loop
                break

            # =========================================
            # FINAL TEXT RESPONSE
            # =========================================

            if (
                hasattr(part, "text")
                and part.text
            ):
                return part.text

        else:
            continue

    # =============================================
    # MAX STEP PROTECTION
    # =============================================

    return (
        "Stopped after reaching the maximum "
        f"reasoning limit ({MAX_STEPS} steps)."
    )