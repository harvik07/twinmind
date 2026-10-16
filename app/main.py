"""TwinMind dashboard.

    streamlit run app/main.py              # live if a key and the local files exist
    streamlit run app/main.py -- --demo    # demo mode regardless

One process, no background workers: the surrogate and the twin load once
(st.cache_resource), and one agent goal or twin verification runs at a time.
Without a Groq key or the local model files every page runs in demo mode on
replayed benchmark results (app/demo_cache/), each one labelled as such.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

from twinmind.ui import components as ui
from twinmind.ui.services import detect_mode

if "--demo" in sys.argv[1:]:
    os.environ["TWINMIND_DEMO"] = "1"

st.set_page_config(page_title="TwinMind", page_icon="🏭", layout="wide")
ui.sidebar(detect_mode())
st.navigation(
    [
        st.Page("views/ask.py", title="Ask", icon="💬", default=True),
        st.Page("views/whatif.py", title="What-if", icon="🎚️"),
        st.Page("views/evidence.py", title="Evidence", icon="📊"),
        st.Page("views/history.py", title="History", icon="🕘"),
    ]
).run()
