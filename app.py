import os
import streamlit as st
import google.generativeai as genai
import tempfile
import pandas as pd
import PyPDF2
from gtts import gTTS
import base64
import requests


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else None)
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY as an environment variable or in Streamlit secrets.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)




st.set_page_config(page_title="Gemini AI Playground", page_icon="🤖", layout="wide")
st.markdown("""
<style>
body, .main .block-container, .stApp {
    background: #181a20 !important;
    color: #e6e6e6 !important;
}
.stMarkdown, .stMarkdown * {
    color: #e6e6e6 !important;
}
.stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #ffe082 !important;
}
.st-emotion-cache-1kyxreq, .st-emotion-cache-1v0mbdj, .st-emotion-cache-1avcm0n {
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}
.stTextArea textarea {
    background: #23262f;
    border-radius: 8px;
    border: 1.5px solid #ffe082;
    font-size: 1.1rem;
    color: #e6e6e6;
}
.stButton>button {
    background: linear-gradient(90deg, #ffe082 0%, #ffd600 100%);
    color: #181a20;
    border-radius: 8px;
    font-weight: 600;
    border: none;
    box-shadow: 0 2px 8px #ffe08244;
    transition: 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #ffd600 0%, #ffe082 100%);
    color: #181a20;
}
.stDataFrame, .stDataFrame table {
    background: #23262f;
    border-radius: 8px;
    font-size: 1.05rem;
    color: #e6e6e6;
}
.stExpanderHeader {
    font-weight: 600;
    color: #ffe082;
}
.stMarkdown code {
    background: #23262f;
    color: #ffe082 !important;
    border-radius: 4px;
    padding: 2px 6px;
}
.stStatus, .stInfo, .stSuccess, .stWarning {
    border-radius: 8px !important;
    background: #23262f !important;
    color: #ffe082 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='display: flex; align-items: center; gap: 1rem;'>
    <span style='font-size:2.5rem;'>🤖</span>
    <span style='font-size:2.1rem; font-weight:700; letter-spacing:-1px;'>Gemini AI Playground</span>
</div>
<div style='color:#888; font-size:1.1rem; margin-bottom:0.5rem;'>A robust, multimodal, and interactive Streamlit app for Google Gemini.</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How to use this app", expanded=False):
    st.markdown("""
    - **Enter a prompt** and select a persona or template to guide the AI.
    - **Upload images** for multimodal input, or a knowledge base for RAG.
    - **Use the sidebar** for advanced controls, moderation, and bonus features.
    - **Interact with CSV data** in the Structured Data Q&A section.
    - **Bonus:** Generate images or audio from AI output!
    """)

prompt_templates = {
    "Write a short story": "Write a short story about a robot who learns to love.",
    "Generate a poem": "Generate a poem about the beauty of the night sky.",
    "Summarize text": "Summarize the following text:",
    "Explain a concept": "Explain the concept of quantum computing in simple terms.",
}
personas = {
    "Creative Writer": "You are a creative and imaginative writer.",
    "Technical Expert": "You are a technical expert who explains things clearly and concisely.",
    "Witty Historian": "You are a witty historian who adds fun facts and humor.",
    "Friendly Assistant": "You are a friendly and helpful assistant."
}

st.sidebar.markdown("---")
st.sidebar.subheader("Custom Persona / Context (Simulated Fine-tuning)")
custom_persona = st.sidebar.text_area(
    "Add custom instructions, facts, or personality traits for the AI (optional):",
    value=st.session_state.get("custom_persona", ""),
    help="This will be prepended to every prompt."
)
if st.sidebar.button("Save Custom Persona"):
    st.session_state["custom_persona"] = custom_persona
    st.sidebar.success("Custom persona/context saved.")


st.sidebar.header("AI Controls")
template_choice = st.sidebar.selectbox("Choose a prompt template", ["(None)"] + list(prompt_templates.keys()))
persona_choice = st.sidebar.radio("Choose an AI persona", list(personas.keys()), index=0)


st.sidebar.markdown("---")
st.sidebar.subheader("Multimodal Input (Optional)")
uploaded_image = st.sidebar.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.markdown("---")
st.sidebar.subheader("Custom Moderation")
blacklist_input = st.sidebar.text_area(
    "Blacklisted keywords/phrases (comma-separated)",
    value=st.session_state.get("blacklist", ""),
    help="Any prompt or output containing these will be flagged."
)
if st.sidebar.button("Save Blacklist"):
    st.session_state["blacklist"] = blacklist_input
    st.sidebar.success("Blacklist updated.")
blacklist = [w.strip().lower() for w in st.session_state.get("blacklist", "").split(",") if w.strip()]

if template_choice != "(None)":
    default_prompt = prompt_templates[template_choice]
else:
    default_prompt = ""

st.sidebar.markdown("---")
st.sidebar.subheader("Knowledge Base (RAG)")
kb_file = st.sidebar.file_uploader("Upload a .txt or .pdf knowledge base", type=["txt", "pdf"])
kb_text = ""
if kb_file:
    if kb_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(kb_file)
        kb_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    else:
        kb_text = kb_file.read().decode("utf-8", errors="ignore")
    st.sidebar.success("Knowledge base loaded.")
    st.sidebar.info(f"Loaded {len(kb_text)} characters.")

with st.container():
    colA, colB = st.columns([3, 1])
    with colA:
        prompt = st.text_area("💬 Enter your prompt:", value=default_prompt, height=120, key="main_prompt")
    with colB:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Higher = more creative")
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=2048, value=512, step=10)
        generate_btn = st.button("🚀 Generate", use_container_width=True)

output_placeholder = st.empty()

st.sidebar.markdown("---")
st.sidebar.subheader("Structured Data (CSV)")
csv_file = st.sidebar.file_uploader("Upload a CSV file for data Q&A", type=["csv"])
df = None
if csv_file:
    df = pd.read_csv(csv_file)
    st.sidebar.success(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    st.sidebar.dataframe(df.head())

if df is not None:
    st.markdown("---")
    with st.expander("📊 Structured Data Q&A (CSV)", expanded=False):
        st.write("Interact with your uploaded CSV data using filters or natural language queries.")
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("Select column to filter", [None] + list(df.columns), key="csv_col")
        with col2:
            filter_value = st.text_input("Filter value (optional)", key="csv_filter")

        nl_query = st.text_input("Ask a question about the data (natural language)", key="csv_nlq")
        run_data_query = st.button("Run Data Query", key="csv_run")

        filtered_df = df
        if selected_col and filter_value:
            filtered_df = df[df[selected_col].astype(str).str.contains(filter_value, case=False, na=False)]
            st.info(f"Filtered by {selected_col} contains '{filter_value}' ({filtered_df.shape[0]} rows)")
        st.dataframe(filtered_df.head(20), use_container_width=True)

        if run_data_query and nl_query.strip():
            st.markdown("---")
            st.subheader("AI Data Q&A")
            st.info(f"Question: {nl_query}")
            code_prompt = (
                f"You are a Python data analyst. Given the following DataFrame columns: {list(df.columns)}. "
                f"Write a Pandas code snippet (no explanation, just code) to answer: '{nl_query}'. "
                f"Assume the DataFrame is named 'df'. Limit output to 20 rows."
            )
            model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
            response = model.generate_content(code_prompt)
            code = response.text.strip()
            import re
            code_match = re.search(r'```(?:python)?\n([\s\S]+?)```', code)
            if code_match:
                code = code_match.group(1)
            st.code(code, language="python")
            try:
                allowed_builtins = {'df': df, 'pd': pd}
                result = eval(code, {"__builtins__": {}}, allowed_builtins)
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result.head(20), use_container_width=True)
                else:
                    st.write(result)
            except Exception as ex:
                st.error(f"Could not execute generated code: {ex}")


def get_weather(city: str):
    weather_db = {
        "london": "Cloudy, 18°C",
        "new york": "Sunny, 25°C",
        "paris": "Rainy, 16°C",
        "tokyo": "Clear, 22°C",
    }
    return weather_db.get(city.lower(), f"Weather data for {city} is not available.")

if generate_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt before generating.")
    elif any(b in prompt.lower() for b in blacklist):
        st.error("Your prompt contains blacklisted keywords/phrases. Please revise your input.")
    else:
        persona_instruction = personas[persona_choice]
        custom_context = st.session_state.get("custom_persona", "")
        full_prompt = ""
        if custom_context:
            full_prompt += f"{custom_context}\n\n"
        full_prompt += f"{persona_instruction}\n\n{prompt.strip()}"

        if kb_text:
            import re
            keywords = [w for w in prompt.strip().split() if len(w) > 3]
            found = []
            for kw in keywords:
                matches = re.findall(rf"(.{{0,60}}{re.escape(kw)}.{{0,60}})", kb_text, re.IGNORECASE)
                found.extend(matches)
            found = list(set(found))[:5]
            if found:
                kb_context = "\n---\n".join(found)
                full_prompt = f"[Knowledge Base Context]\n{kb_context}\n\n{full_prompt}"
                st.info("Relevant knowledge base context injected into prompt.")

        if df is not None and any(x in prompt.lower() for x in ["csv", "table", "data", "row", "column", "pandas", "dataframe"]):
            sample = df.head(10).to_csv(index=False)
            full_prompt = f"[CSV Data Sample]\n{sample}\n\n{full_prompt}"
            st.info("CSV data sample injected into prompt.")

        tool_call_result = None
        tool_call_city = None
        import re
        weather_pattern = re.compile(r"weather in ([a-zA-Z ]+)", re.IGNORECASE)
        match = weather_pattern.search(prompt)
        if match:
            tool_call_city = match.group(1).strip()
            with st.status("AI is trying to fetch weather data...", expanded=True) as status:
                st.info(f"Tool call: get_weather('{tool_call_city}')")
                tool_call_result = get_weather(tool_call_city)
                st.success(f"Result: {tool_call_result}")
                status.update(label="Weather data fetched.", state="complete")

        def multi_step_reasoning(prompt):
            steps = [
                ("Step 1: Analyze intent", lambda p, _: f"Analyze the following user request and break it into logical steps: {p}"),
                ("Step 2: Generate initial draft", lambda p, prev: f"Write a detailed draft response for the following user request, using this plan: {prev}"),
                ("Step 3: Refine output", lambda p, prev: f"Refine and clarify the following draft for the user request '{p}':\n{prev}"),
            ]
            outputs = []
            prev_output = None
            for i, (label, step_fn) in enumerate(steps):
                with st.status(f"{label}...", expanded=True) as status:
                    step_prompt = step_fn(prompt, prev_output)
                    st.info(f"Prompt: {step_prompt}")
                    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
                    response = model.generate_content(step_prompt)
                    step_output = response.text
                    with st.expander(f"{label} Output", expanded=(i==len(steps)-1)):
                        st.write(step_output)
                    status.update(label=f"{label} complete.", state="complete")
                    outputs.append(step_output)
                    prev_output = step_output
            return outputs[-1]

        with st.spinner("Generating response..."):
            if any(x in prompt.lower() for x in ["multi-step", "chain", "reasoning", "step by step"]):
                output_text = multi_step_reasoning(full_prompt)
            else:
                import time
                if uploaded_image:
                    import PIL.Image
                    from io import BytesIO
                    image = PIL.Image.open(BytesIO(uploaded_image.read()))
                    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
                    multimodal_prompt = full_prompt
                    if tool_call_result:
                        multimodal_prompt = f"Weather info: {tool_call_result}\n\n{full_prompt}"
                    response = model.generate_content(
                        [multimodal_prompt, image],
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": int(max_tokens),
                        },
                    )
                else:
                    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
                    text_prompt = full_prompt
                    if tool_call_result:
                        text_prompt = f"Weather info: {tool_call_result}\n\n{full_prompt}"
                    response = model.generate_content(
                        text_prompt,
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": int(max_tokens),
                        },
                        stream=True
                    )
                output_text = ""
                progress = st.progress(0, text="Streaming Gemini output...")
                chunk_count = 0
                if hasattr(response, "__iter__"):
                    for chunk in response:
                        chunk_count += 1
                        output_text += getattr(chunk, "text", str(chunk))
                        output_placeholder.markdown(f"**Gemini Output:**\n\n{output_text}")
                        progress.progress(min(chunk_count * 5, 100), text="Streaming Gemini output...")
                        time.sleep(0.01)
                elif hasattr(response, "text"):
                    for word in response.text.split():
                        output_text += word + " "
                        output_placeholder.markdown(f"**Gemini Output:**\n\n{output_text}")
                        chunk_count += 1
                        progress.progress(min(chunk_count * 2, 100), text="Streaming Gemini output...")
                        time.sleep(0.01)
                else:
                    output_text = str(response)
                    output_placeholder.markdown(f"**Gemini Output:**\n\n{output_text}")
                progress.progress(100, text="Streaming complete.")

            if any(b in output_text.lower() for b in blacklist):
                st.warning("AI output contains blacklisted keywords/phrases. Please review the content.")

            import json
            displayed = False
            # JSON
            if any(x in prompt.lower() for x in ["json", "list of", "table", "structured", "dictionary", "summarize", "summary", "extract", "parse"]):
                try:
                    json_start = output_text.find('{')
                    json_end = output_text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = output_text[json_start:json_end]
                        parsed = json.loads(json_str)
                        output_placeholder.json(parsed)
                        displayed = True
                except Exception:
                    pass
            if not displayed and '|' in output_text and output_text.count('|') > 2:
                import re
                table_match = re.search(r'(\|.+\|\n\|[\s\S]+?\|\n)', output_text)
                if table_match:
                    st.markdown(table_match.group(1))
                    displayed = True
            if not displayed:
                output_placeholder.markdown(f"**Gemini Output:**\n\n{output_text}")

            prompt_tokens = max(1, len(full_prompt) // 4)
            response_tokens = max(1, len(output_text) // 4)
            total_tokens = prompt_tokens + response_tokens
            cost_per_1k = 0.000125
            estimated_cost = (total_tokens / 1000) * cost_per_1k
            st.markdown(f"""
<div style='background:#23262f; color:#ffe082; border-radius:8px; padding:10px 16px; font-size:1.08rem; border:1.5px solid #ffe082; margin-bottom:0.5rem;'>
<b>Estimated tokens used:</b> <span style='color:#fffde7'>{total_tokens}</span> <span style='color:#bdbdbd;'>(Prompt: {prompt_tokens}, Response: {response_tokens})</span><br>
<b>Estimated cost:</b> <span style='color:#fffde7'>${estimated_cost:.6f}</span>
</div>
""", unsafe_allow_html=True)

            st.session_state["last_output_text"] = output_text
            st.session_state["last_prompt"] = prompt.strip()
    st.session_state["show_bonus_buttons"] = True

if st.session_state.get("show_bonus_buttons"):
    st.markdown("---")
    st.subheader("Bonus Features")
    col1, col2 = st.columns(2)
    with col1:
        imggen_clicked = st.button("Generate Image from Response", key="imggen")
    with col2:
        ttsgen_clicked = st.button("Generate Audio from Response", key="ttsgen")

    st.markdown("---")
    st.subheader("Conversation History")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    last_output = st.session_state.get("last_output_text", "")
    last_prompt = st.session_state.get("last_prompt", "")
    if last_output:
        import json
        displayed = False
        if any(x in last_prompt.lower() for x in ["json", "list of", "table", "structured", "dictionary", "summarize", "summary", "extract", "parse"]):
            try:
                json_start = last_output.find('{')
                json_end = last_output.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = last_output[json_start:json_end]
                    parsed = json.loads(json_str)
                    st.json(parsed)
                    displayed = True
            except Exception:
                pass
        if not displayed:
            st.markdown(f"**Gemini Output:**\n\n{last_output}")

        if st.session_state.get("last_generated_image"):
            import base64
            st.image(base64.b64decode(st.session_state["last_generated_image"]), caption="Generated Image", use_container_width=True)
        if st.session_state.get("last_generated_audio"):
            st.audio(st.session_state["last_generated_audio"], format="audio/mp3")

    if imggen_clicked:
        with st.spinner("Generating image from text using Stability AI..."):
            stability_api_key = "sk-A4CbJHxRTpefmasipb3JNdODGjX49Q4OPNTzqf9r7zK3CMGg"
            api_url = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image"
            headers = {
                "Authorization": f"Bearer {stability_api_key}",
                "Content-Type": "application/json"
            }
            prompt_text = st.session_state.get("last_prompt", "AI generated image")
            import string
            if not prompt_text or not prompt_text.strip() or len(prompt_text.strip()) < 5 or all(c in string.punctuation for c in prompt_text.strip()):
                prompt_text = "AI generated image"
            else:
                prompt_text = prompt_text.strip()[:2000]
            payload = {
                "text_prompts": [{"text": prompt_text}],
                "cfg_scale": 7,
                "clip_guidance_preset": "FAST_BLUE",
                "height": 128,
                "width": 128,
                "samples": 1,
                "steps": 30
            }
            try:
                r = requests.post(api_url, json=payload, headers=headers, timeout=60)
                if r.status_code == 200:
                    response_json = r.json()
                    if "artifacts" in response_json and len(response_json["artifacts"]) > 0:
                        img_b64 = response_json["artifacts"][0]["base64"]
                        st.session_state["last_generated_image"] = img_b64
                        st.image(base64.b64decode(img_b64), caption="Generated Image (128x128)", width=128)
                    else:
                        st.error("Stability AI did not return any images.")
                        st.session_state["last_generated_image"] = None
                elif r.status_code == 400:
                    st.error("Stability AI: Bad request. The prompt must be 1-2000 characters. Please try a shorter or non-empty prompt.")
                elif r.status_code == 401:
                    st.error("Stability AI: Unauthorized. Please check your API key.")
                elif r.status_code == 403:
                    st.error("Stability AI: Access forbidden. Check your account limits or API key.")
                elif r.status_code == 404:
                    st.error("Stability AI: The specified engine was not found. Please check the engine name or your API access.")
                elif r.status_code == 429:
                    st.error("Stability AI: Rate limit exceeded. Try again later.")
                else:
                    st.error(f"Image generation failed. Status: {r.status_code}, Response: {r.text}")
            except Exception as ex:
                st.error(f"Image generation error: {ex}")
                st.session_state["last_generated_image"] = None

    if ttsgen_clicked:
        with st.spinner("Generating audio..."):
            try:
                output_text = st.session_state["last_output_text"]
                if not output_text.strip():
                    st.warning("No output text to convert to audio.")
                elif len(output_text) > 4000:
                    st.warning("Text too long for TTS. Please try a shorter response.")
                else:
                    tts = gTTS(output_text)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        tts.save(fp.name)
                    with open(fp.name, "rb") as f:
                        audio_bytes = f.read()
                    st.session_state["last_generated_audio"] = audio_bytes
                    st.audio(audio_bytes, format="audio/mp3")
            except Exception as ex:
                st.error(f"Audio generation error: {ex}")
                st.session_state["last_generated_audio"] = None
