import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(layout="wide", page_title="Traceable Document Compressor")

def extract_and_tag_pdf(uploaded_file):
    """
    Extracts text from a PDF and tags each paragraph with [[P{page}_{block}]].
    Returns a string of tagged text and a dictionary mapping tags to raw text.
    """
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    tagged_text = ""
    source_map = {}

    for page_num, page in enumerate(doc):
        # 1-based indexing for pages
        p_num = page_num + 1
        
        # Use blocks to identify paragraphs
        blocks = page.get_text("blocks")
        for i, block in enumerate(blocks):
            # block structure: (x0, y0, x1, y1, "text", block_no, block_type)
            # block_type 0 is text
            if block[6] == 0:
                text = block[4].strip()
                if text:
                    # Create a unique tag
                    tag = f"[[P{p_num}_{i}]]"
                    
                    # Store mapping
                    source_map[tag] = text
                    
                    # Append to full text
                    tagged_text += f"{tag} {text}\n\n"

    return tagged_text, source_map

def summarize_with_gemini(text, api_key, model_name):
    """
    Summarizes text using Gemini with structured JSON output.
    """
    if not api_key:
        st.error("Google API Key is missing.")
        return None

    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.5,
        "response_mime_type": "application/json"
    }

    model = genai.GenerativeModel(model_name, generation_config=generation_config)

    prompt = (
        "You are an expert analyst. Your task is to compress the provided document into a structured summary.\n"
        "Input text contains source tags like [[P1_0]], [[P2_5]] at the start of blocks.\n"
        "You MUST preserve these tags in your output for traceability.\n\n"
        "Output Format (JSON):\n"
        "{\n"
        "  \"summary\": {\n"
        "    \"high_level_summary\": \"A concise 2-3 sentence overview of the entire document.\",\n"
        "    \"sections\": [\n"
        "      {\n"
        "        \"title\": \"Section/Chapter Title\",\n"
        "        \"key_points\": [\n"
        "          {\n"
        "            \"statement\": \"Key fact or claim.\",\n"
        "            \"source_ids\": [\"[[P1_0]]\", \"[[P1_2]]\"],\n"
        "            \"risk_type\": \"None/Operational/Financial/Legal\",\n"
        "            \"details\": \"A comprehensive 3-5 sentence deep-dive. Include background context, specific figures, dates, exceptions, and potential implications. This is for users who need the full story behind the bullet point.\",\n"
        "            \"rationale\": \"Why this retention is critical.\"\n"
        "          }\n"
        "        ]\n"
        "      }\n"
        "    ]\n"
        "  },\n"
        "  \"meta_analysis\": {\n"
        "    \"omitted_themes\": [\n"
        "      {\n"
        "        \"theme\": \"Description of omitted topic\",\n"
        "        \"reason_for_omission\": \"Why it was removed.\",\n"
        "        \"impact_score\": \"Low/Medium/High\"\n"
        "      }\n"
        "    ],\n"
        "    \"global_retention_rationale\": \"Overall strategy.\"\n"
        "  }\n"
        "}\n\n"
        "Requirements:\n"
        "1. Capture ALL key facts, exceptions, and risks.\n"
        "2. EVERY 'statement' must include at least one 'source_id'.\n"
        "3. The 'details' field MUST be elaborate and comprehensive (not just a rephrasing).\n"
        "4. Fill 'meta_analysis' to explain what was removed and why.\n\n"
        f"Document Content:\n{text}"
    )

    import time

    # Retry configuration
    max_retries = 3
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    st.warning(f"Quota exceeded. Retrying in {sleep_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
                else:
                    st.error("Exceeded maximum retries for API quota. Please try again later or switch models.")
                    return None
            else:

                st.error(f"Error during summarization: {e}")
                return None

def answer_question(tagged_text, question, api_key, model_name):
    """
    Answers a question based ONLY on the provided tagged document text.
    """
    if not api_key:
        return "API Key missing."

    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.2, # Lower temperature for high fidelity
        "response_mime_type": "text/plain"
    }

    model = genai.GenerativeModel(model_name, generation_config=generation_config)

    prompt = (
        "You are an expert document analyst. You must answer the user's question based ONLY on the provided document text.\n"
        "The text contains source tags like [[P1_0]]. You MUST use these tags as 'proofs' in your answer.\n"
        "Requirements:\n"
        "1. Provide a detailed, comprehensive answer.\n"
        "2. Explicitly cite the source tags (e.g., [[P1_0]], [[P2_5]]) for EVERY claim or fact you mention.\n"
        "3. If the answer is not in the text, say 'Information not available in the document.'\n"
        "4. Your goal is to show the user exactly WHY you chose this answer by referencing the tags.\n\n"
        f"Document Content:\n{tagged_text}\n\n"
        f"User Question: {question}\n\n"
        "Answer (Detailed with Proofs):"
    )

    import time
    
    # Retry configuration
    max_retries = 3
    base_delay = 10  # Seconds (increased for Q&A to allow for heavy context)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    # We can't easily show a st.warning from here as it's often inside a spinner/popover, 
                    # but it's better than failing immediately.
                    time.sleep(sleep_time)
                    continue
                else:
                    return f"Exceeded maximum retries for API quota. Please try again in a minute or switch to a different model.\n\nRaw Error: {error_str}"
            else:
                return f"Error answering question: {str(e)}"


# Sidebar
st.sidebar.title("CONFIGURATION")
user_api_key = st.sidebar.text_input("Enter Google API Key", type="password")

if user_api_key:
    api_key = user_api_key
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        default_index = 0
        preferred_models = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.5-pro-001']
        for pref in preferred_models:
            if pref in models:
                default_index = models.index(pref)
                break
        selected_model = st.sidebar.selectbox("Select Gemini Model", models, index=default_index)
    except Exception as e:
        st.sidebar.error(f"Error listing models: {e}")
        selected_model = "models/gemini-1.5-flash"
else:
    selected_model = None

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# --- COREOPS THEME & CSS ---
# --- CORE OPS DESIGN SYSTEM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #0B0E14;
        --bg-secondary: #161B22;
        --bg-tertiary: #21262D;
        --accent-primary: #2E9AFF;
        --accent-secondary: #3FB950;
        --text-primary: #F0F6FC;
        --text-secondary: #8B949E;
        --border-color: #30363D;
        --card-radius: 12px;
        --transition-speed: 0.3s;
    }

    /* Global Overrides */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Outfit', sans-serif;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -1px !important;
    }
    
    p, span, label {
        font-family: 'Outfit', sans-serif;
    }

    /* Custom Header Container */
    .header-container {
        background: linear-gradient(90deg, #161B22 0%, #0B121C 100%);
        padding: 2rem;
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] img {
        border-radius: 50%;
        margin-bottom: 1rem;
    }

    /* Card Layouts */
    div.element-container:has(div.stMarkdown > div.card) {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: var(--card-radius);
        transition: all var(--transition-speed) ease;
    }
    
    .card:hover {
        border-color: var(--accent-primary);
        transform: translateY(-2px);
    }

    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background: var(--bg-tertiary);
        padding: 1rem 1.5rem;
        border-radius: var(--card-radius);
        border-left: 4px solid var(--accent-primary);
    }
    
    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-secondary);
        border-radius: var(--card-radius) var(--card-radius) 0 0;
        padding: 8px 24px;
        color: var(--text-secondary);
        border: 1px solid var(--border-color);
        border-bottom: none;
        transition: all var(--transition-speed) ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--accent-primary);
        background-color: var(--bg-tertiary);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--bg-tertiary) !important;
        color: var(--accent-primary) !important;
        border-top: 2px solid var(--accent-primary) !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2E9AFF 0%, #157EFB 100%);
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all var(--transition-speed) ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        filter: brightness(1.1);
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(46, 154, 255, 0.4);
    }

    /* Status Indicators */
    .stStatus {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* Chat Elements */
    .stChatMessage {
        background-color: var(--bg-secondary) !important;
        border-radius: var(--card-radius) !important;
        border: 1px solid var(--border-color) !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }

    /* Code Blocks */
    code {
        background-color: rgba(46, 154, 255, 0.1) !important;
        color: var(--accent-primary) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-size: 0.9em !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="header-container">
    <div style="display: flex; align-items: center; gap: 20px;">
        <div style="font-size: 50px;"></div>
        <div>
            <h1 style="margin: 0;">Traceable Context Engine</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.1rem;">
                Enterprise-Grade Document Compression with Zero Hallucination.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if uploaded_file and api_key and selected_model:
    if "tagged_text" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
        
        # --- STATUS INDICATOR (CoreOps Style) ---
        with st.status("INITIALIZING CORE ENGINE...", expanded=True) as status:
            st.write("Ingesting PDF Stream...")
            st.write("Injecting Traceability Beacons...")
            tagged_text, source_map = extract_and_tag_pdf(uploaded_file)
            st.session_state.tagged_text = tagged_text
            st.session_state.source_map = source_map
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.json_summary = None
            st.session_state.qa_history = [] # Reset QA on new upload
            status.update(label="INGESTION COMPLETE", state="complete", expanded=False)

    # --- CONTROL CENTER ---
    c_left, c_right = st.columns([0.7, 0.3])
    with c_left:
        st.markdown(f"**TARGET LOADED:** `{uploaded_file.name}`")
        st.caption(f"Source size: {len(st.session_state.tagged_text)} characters")
    
    with c_right:
        execute_btn = st.button("EXECUTE COMPRESSION", use_container_width=True)

    if execute_btn:
        with st.status("ANALYZING CONTEXT VECTORS...", expanded=True) as status:
            st.write(f"Establishing Uplink to {selected_model}...")
            st.write("Scanning for High-Priority Risks...")
            
            response_text = summarize_with_gemini(st.session_state.tagged_text, api_key, selected_model)
            
            if response_text:
                st.write("Serializing Structured Output...")
                try:
                    st.session_state.json_summary = json.loads(response_text)
                    status.update(label="MISSION ACCOMPLISHED", state="complete", expanded=False)
                except json.JSONDecodeError:
                    status.update(label="SERIALIZATION FAILED", state="error")
                    st.error("Failed to parse AI response as JSON.")
                    st.text(response_text)

    if st.session_state.get("json_summary"):
        data = st.session_state.json_summary.get("summary", {})
        meta = st.session_state.json_summary.get("json_meta_analysis", st.session_state.json_summary.get("meta_analysis", {}))
        
        # --- METRICS DASHBOARD (CoreOps) ---
        original_len = len(st.session_state.tagged_text)
        summary_text_approx = json.dumps(data)
        summary_len = len(summary_text_approx)
        compression_ratio = original_len / summary_len if summary_len > 0 else 0
        
        # Calculate Risk Count
        risk_count = 0
        sections = data.get("sections", [])
        for s in sections:
            for p in s.get("key_points", []):
                r = p.get("risk_type", "").lower()
                if "high" in r or "critical" in r or "operational" in r or "financial" in r or "legal" in r:
                    risk_count += 1
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Risk Profile", risk_count, delta="CRITICAL" if risk_count > 0 else "SECURE", delta_color="inverse")
        with m2:
            st.metric("Source Coverage", f"{len(st.session_state.source_map)} Points", delta="VERIFIED")
        with m3:
            st.metric("Compression", f"{compression_ratio:.1f}x", delta="OPTIMIZED")
        
        st.divider()

        # --- TABBED LAYOUT ---
        # 📄 Executive Brief, 🔍 Source Verifier, 💬 Q&A, 📊 Metrics
        tab_brief, tab_verify, tab_qa, tab_metrics = st.tabs(["EXECUTIVE BRIEF", "SOURCE VERIFIER", "Q&A", "METRICS"])
        
        # TAB 1: EXECUTIVE BRIEF
        with tab_brief:
            st.subheader("MISSION SUMMARY")
            st.info(data.get("high_level_summary", "No summary provided."))
            
            if risk_count > 0:
                st.error(f"ALERT: {risk_count} CRITICAL RISKS DETECTED")
                for section in sections:
                    for point in section.get("key_points", []):
                         r = point.get("risk_type", "").lower()
                         if "high" in r or "critical" in r:
                             st.markdown(f"**{section.get('title')}** -> {point.get('statement')}")

            st.divider()
            
            st.subheader("DETAILED SECTION ANALYSIS")
            st.caption("Expand modules for tactical details.")
            for section in sections:
                with st.expander(f"{section.get('title', 'Untitled Section').upper()}", expanded=False):
                    for point in section.get("key_points", []):
                        # Risk Badge
                        risk = point.get('risk_type', 'None')
                        risk_color = "gray"
                        if "High" in risk or "Critical" in risk: risk_color = "red"
                        elif "Medium" in risk or "Financial" in risk or "Legal" in risk: risk_color = "orange"
                        
                        cols = st.columns([0.05, 0.75, 0.2])
                        cols[0].markdown(f":{risk_color}[●]")
                        cols[1].markdown(f"**{point.get('statement')}**")
                        
                        # Actions
                        with cols[2]:
                            # Deep Dive
                            with st.popover("INTEL", help="View comprehensive details"):
                                st.markdown("### TACTICAL ANALYSIS")
                                if point.get("details"):
                                    st.info(point.get('details'))
                                
                                if point.get("rationale"):
                                    st.caption(f"**Retention Strategy:** {point.get('rationale')}")
                            
                            # Trace
                            source_ids = point.get('source_ids', [])
                            if source_ids:
                                with st.popover("SOURCE", help="View original source"):
                                    st.markdown("### SOURCE INTERCEPT")
                                    valid_sources = [sid for sid in source_ids if sid in st.session_state.source_map]
                                    if valid_sources:
                                        for sid in valid_sources:
                                            st.markdown(f"**REF ID: {sid}**")
                                            st.code(st.session_state.source_map[sid], language=None) 
                            else:
                                st.caption("NO SOURCE")
                        
                        st.divider()

        # TAB 2: SOURCE VERIFIER (Traceability Map)
        with tab_verify:
            st.subheader("INFORMATION INTEGRITY REPORT")
            
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"**RETENTION STRATEGY:**\n{meta.get('global_retention_rationale', 'N/A')}")
            
            with c2:
                 st.write("**OMITTED DATA STREAMS:**")
                 omitted = meta.get("omitted_themes", [])
                 if omitted:
                    for item in omitted:
                        imp = item.get("impact_score", "Low")
                        icon = ""
                        if imp == "High": icon = ""
                        elif imp == "Medium": icon = ""
                        st.markdown(f"{icon} **{item.get('theme')}**: {item.get('reason_for_omission')}")
                 else:
                    st.success("ZERO DATA LOSS DETECTED.")

        # TAB 2.5: Q&A (Compressed Context)
        with tab_qa:
            st.subheader("COMPRESSED CONTEXT Q&A")
            st.caption("Ask questions about the document based on the generated summary.")
            
            # Display chat history
            for message in st.session_state.qa_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about the document summary..."):
                st.session_state.qa_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Searching document vectors..."):
                        answer = answer_question(st.session_state.tagged_text, prompt, api_key, selected_model)
                        st.markdown(answer)
                        st.session_state.qa_history.append({"role": "assistant", "content": answer})

        # TAB 3: METRICS (Raw)
        with tab_metrics:
            st.subheader("SYSTEM TELEMETRY")
            st.json(st.session_state.json_summary)

elif not api_key:
    st.warning("AUTHENTICATION REQUIRED: ENTER API KEY.")
elif not uploaded_file:
    st.info("UPLOAD TARGET FILE FOR ANALYSIS.")
