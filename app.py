import os
import streamlit as st
import google.generativeai as genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else None)
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY as an environment variable or in Streamlit secrets.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)



st.set_page_config(page_title="Gemini Text Generation Demo", page_icon="🤖")
st.title("🤖 Gemini Text Generation Demo")
st.markdown("""
Welcome! Enter a prompt and let Google's Gemini model generate a response.  
**Temperature** controls creativity (lower = more focused, higher = more creative).  
**Max Output Tokens** controls the length of the response.
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
    "Friendly Assistant": "You are a friendly and helpful assistant.",
}

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

prompt = st.text_area("Enter your prompt:", value=default_prompt, height=120, key="main_prompt")
temperature = st.slider("Temperature (creativity)", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Higher values = more creative, lower = more focused.")
max_tokens = st.number_input("Max Output Tokens", min_value=50, max_value=2048, value=512, step=10)
generate_btn = st.button("Generate")

st.markdown("---")
st.subheader("Conversation History")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

output_placeholder = st.empty()


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
        full_prompt = f"{persona_instruction}\n\n{prompt.strip()}"

        tool_call_result = None
        tool_call_city = None
        import re
        weather_pattern = re.compile(r"weather in ([a-zA-Z ]+)", re.IGNORECASE)
        match = weather_pattern.search(prompt)
        if match:
            tool_call_city = match.group(1).strip()
            tool_call_result = get_weather(tool_call_city)
            with st.status("AI is trying to fetch weather data...", expanded=True) as status:
                st.info(f"Tool call: get_weather('{tool_call_city}')")
                st.success(f"Result: {tool_call_result}")
                status.update(label="Weather data fetched.", state="complete")

        try:
            with st.spinner("Generating response..."):
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
                    )
                output_text = ""
                if hasattr(response, "text"):
                    if hasattr(response, "__iter__") and not isinstance(response.text, str):
                        for chunk in response:
                            output_text += chunk.text
                            output_placeholder.markdown(f"**Gemini Output:**\n\n{output_text}")
                    else:
                        for word in response.text.split():
                            output_text += word + " "
                            output_placeholder.markdown(f"**Gemini Output:**\n\n{output_text}")
                else:
                    output_text = str(response)
                    output_placeholder.markdown(f"**Gemini Output:**\n\n{output_text}")

                if any(b in output_text.lower() for b in blacklist):
                    st.warning("AI output contains blacklisted keywords/phrases. Please review the content.")

                import json
                displayed = False
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
                if not displayed:
                    output_placeholder.markdown(f"**Gemini Output:**\n\n{output_text}")

                prompt_tokens = max(1, len(full_prompt) // 4)
                response_tokens = max(1, len(output_text) // 4)
                total_tokens = prompt_tokens + response_tokens
                cost_per_1k = 0.000125
                estimated_cost = (total_tokens / 1000) * cost_per_1k
                st.info(f"Estimated tokens used: {total_tokens} (Prompt: {prompt_tokens}, Response: {response_tokens})\nEstimated cost: ${estimated_cost:.6f}")

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": prompt.strip()
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": output_text
                })
        except Exception as e:
            st.error(f"Error: {e}")
