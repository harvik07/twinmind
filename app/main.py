"""TwinMind dashboard.

    streamlit run app/main.py              # live if a key and the local files exist
    streamlit run app/main.py -- --demo    # demo mode regardless

One process. The models load once (st.cache_resource). A plan is worked out
in ONE background thread, one at a time, so clicking around never stops it;
every page shows a banner with Cancel while it runs. Without a key or the
local model files every page runs in demo mode on saved example results, each
labelled as such.
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
ui.running_banner()
st.navigation(
    [
        st.Page("views/plan.py", title="Plan", icon="🧭", default=True),
        st.Page("views/try_changes.py", title="Try changes", icon="🎚️"),
        st.Page("views/reliability.py", title="How reliable is this?", icon="🔎"),
        st.Page("views/past_results.py", title="Past results", icon="🕘"),
    ]
).run()
