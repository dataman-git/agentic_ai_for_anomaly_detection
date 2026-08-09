import streamlit as st
import os
import sys
import pandas as pd


# Add project root to Python path
ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT_DIR)

from google import genai
from engine.agent_loop import run_agent
from dotenv import load_dotenv

# =====================================================
# Load the api key
# =====================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.write(
    "Key loaded:",
    GEMINI_API_KEY is not None
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Enterprise Agentic AI",
    layout="centered"
)

st.title("Enterprise Agentic AI")
st.caption(
    "Agent with tools + multi-step reasoning"
)


# =====================================================
# SAFE CLIENT INIT
# =====================================================

@st.cache_resource
def init_client():

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


try:

    client = init_client()

except Exception as e:

    st.error(
        f"Client init failed: {e}"
    )

    st.stop()


# =====================================================
# LOAD SKILLS
# =====================================================

def load_skills():

    txt = ""

    skills_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "skills"
    )

    if not os.path.exists(skills_path):
        return ""

    for f in os.listdir(skills_path):

        if f.endswith(".md"):

            with open(
                os.path.join(
                    skills_path,
                    f
                ),
                "r",
                encoding="utf-8"
            ) as file:

                txt += (
                    file.read()
                    + "\n\n"
                )

    return txt


SYSTEM_PROMPT = (
    "Agent with skills:\n"
    + load_skills()
)

# =====================================================
# SESSION STATE
# =====================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "scored_df" not in st.session_state:
    st.session_state.scored_df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "model_name" not in st.session_state:
    st.session_state.model_name = None

if "threshold" not in st.session_state:
    st.session_state.threshold = None

if "X_processed" not in st.session_state:
    st.session_state.X_processed = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# OPTIONAL DATASET STATUS PANEL
# =====================================================

with st.sidebar:

    st.subheader("Session Status")

    if st.session_state.df is not None:

        st.success("Dataset Loaded")

        st.write(
            f"Rows: {len(st.session_state.df)}"
        )

        st.write(
            f"Columns: {st.session_state.df.shape[1]}"
        )

    else:

        st.warning(
            "No Dataset Loaded"
        )

    if st.session_state.scored_df is not None:

        st.success(
            "Model Results Available"
        )


# =====================================================
# CHAT INPUT
# =====================================================

if prompt := st.chat_input(
    "Ask your agent..."
):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    try:

        answer = run_agent(
            prompt=prompt,
            client=client,
            state=st.session_state,
            system_prompt=SYSTEM_PROMPT
        )

    except Exception as e:

        err = str(e)

        if (
            "503" in err
            or "UNAVAILABLE" in err
        ):

            answer = (
                "Model busy. "
                "Try again shortly."
            )

        else:

            answer = (
                f"Error:\n\n{err}"
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# =====================================================
# CHAT HISTORY
# =====================================================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )


# =====================================================
# OPTIONAL DATAFRAME DISPLAY
# =====================================================

if st.session_state.df is not None:

    with st.expander(
        "Current Dataset Preview"
    ):

        st.dataframe(
            st.session_state.df.head(20),
            use_container_width=True
        )


# =====================================================
# OPTIONAL ANOMALY RESULTS DISPLAY
# =====================================================

if st.session_state.scored_df is not None:

    with st.expander(
        "Scored Dataset Preview"
    ):

        st.dataframe(
            st.session_state.scored_df.head(20),
            use_container_width=True
        )