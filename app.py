import streamlit as st
import os
import time
import json
from PIL import Image
import io

try:
    import google.generativeai as genai
except ImportError:
    st.error("google-generativeai is not installed. Please install it with 'pip install google-generativeai'.")

API_KEY = os.getenv("AIzaSyBnYol7F15XN7x6Um5H-VD0o23H8VqCjGk") or st.secrets.get("AIzaSyBnYol7F15XN7x6Um5H-VD0o23H8VqCjGk", None)
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY as an environment variable or in Streamlit secrets.")

def get_weather(city):
    weather_data = {
        "London": {"temp": 20, "condition": "Cloudy"},
        "Paris": {"temp": 25, "condition": "Sunny"},
        "New York": {"temp": 28, "condition": "Rainy"},
    }
    return weather_data.get(city, {"temp": "N/A", "condition": "Unknown"})

def gemini_respond(user_input):
    if "weather" in user_input.lower():
        city = None
        for c in ["London", "Paris", "New York"]:
            if c.lower() in user_input.lower():
                city = c
                break
        if city:
            yield {"type": "function_call", "content": f"AI is trying to fetch weather data for {city}..."}
            weather = get_weather(city)
            yield {"type": "function_result", "content": weather}
            response = f"The current weather in {city} is {weather['temp']}°C and {weather['condition']}. Let me know if you need more details!"
            for word in response.split():
                yield {"type": "stream", "content": word + " "}
                time.sleep(0.07)
            yield {"type": "done"}
            return
    if "show me a table" in user_input.lower():
        table_md = """| City      | Temp (°C) | Condition |
|-----------|-----------|-----------|
| London    | 20        | Cloudy    |
| Paris     | 25        | Sunny     |
| New York  | 28        | Rainy     |"""
        yield {"type": "structured", "content": table_md}
        yield {"type": "done"}
        return
    response = "I'm not sure how to help with that, but I can answer questions about the weather in London, Paris, or New York."
    for word in response.split():
        yield {"type": "stream", "content": word + " "}
        time.sleep(0.07)
    yield {"type": "done"}

def estimate_token_usage(text):
    return max(1, len(text) // 4)

personas = {
    "Creative Writer": "You are a creative writer. Your responses should be imaginative and engaging.",
    "Technical Expert": "You are a technical expert. Your responses should be accurate and informative.",
    "Witty Historian": "You are a witty historian. Your responses should be insightful and entertaining, with a touch of humor.",
    "Neutral": ""
}
prompt_templates = {
    "Select a template": "",
    "Write a short story about...": "Write a short story about ",
    "Generate a poem about...": "Generate a poem about ",
    "Explain this concept:": "Explain this concept in simple terms: ",
    "Translate to French:": "Translate the following to French: "
}

def main():
    st.set_page_config(page_title="Gemini All-in-One Streamlit Demo", layout="wide")

    with st.sidebar:
        st.title("Gemini AI Demo")
        st.info("Try asking: 'What's the weather in Paris?' or 'Show me a table of weather.'")
        st.markdown("---")
        st.write("**Resource Usage Awareness**")
        st.write("Token usage and cost will be shown after each response.")

    st.title("🌟 Gemini All-in-One Streamlit Demo")

    with st.expander("About this app", expanded=False):
        st.markdown("""
        - **Stage 1:** Basic text generation with temperature and token controls.
        - **Stage 2:** Prompt templates, personas, chat history, and multimodal (image) input.
        - **Stage 3:** Function calling, structured output, streaming, and resource usage display.
        """)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    colA, colB = st.columns([2,1])
    with colA:
        selected_persona_name = st.selectbox("Choose a Persona:", list(personas.keys()), key="persona")
        system_instruction = personas[selected_persona_name]

        selected_template = st.selectbox("Use a Prompt Template:", list(prompt_templates.keys()), key="template")
        user_input = st.text_area("Enter your prompt:", value=prompt_templates[selected_template], height=120, key="main_prompt")

        uploaded_file = st.file_uploader("Upload an image (optional):", type=["jpg", "jpeg", "png"])
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.01)
        st.info("Controls the randomness of the output. Higher values result in more creative and diverse text.")

        max_output_tokens = st.number_input("Max Output Tokens", 50, 2048, 500, 1)
        st.info("Sets the maximum number of tokens (words or word pieces) in the generated output.")

        if st.button("Generate", key="generate"):
            if not user_input and image is None:
                st.warning("Please enter a prompt or upload an image.")
            else:
                content = []
                if system_instruction:
                    content.append({"role": "user", "parts": [{"text": system_instruction}]})
                    content.append({"role": "model", "parts": [{"text": "Okay, I understand."}]})
                for message in st.session_state.chat_history:
                    content.append({"role": message["role"], "parts": message["parts"]})
                img_byte_arr = None
                if image:
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="PNG")
                    img_byte_arr = img_byte_arr.getvalue()
                    content.append({"role": "user", "parts": [{"mime_type": "image/png", "data": img_byte_arr}, {"text": user_input}]})
                elif user_input:
                    content.append({"role": "user", "parts": [{"text": user_input}]})

                with st.spinner("Gemini is thinking..."):
                    if not image and ("weather" in user_input.lower() or "show me a table" in user_input.lower()):
                        container = st.empty()
                        stream_text = ""
                        for chunk in gemini_respond(user_input):
                            if chunk["type"] == "function_call":
                                container.info(chunk["content"])
                            elif chunk["type"] == "function_result":
                                container.success(f"Weather data: {chunk['content']}")
                            elif chunk["type"] == "structured":
                                container.markdown(chunk["content"])
                            elif chunk["type"] == "stream":
                                stream_text += chunk["content"]
                                container.markdown(stream_text)
                            elif chunk["type"] == "done":
                                break
                        st.session_state.chat_history.append({"role": "user", "parts": [{"text": user_input}]})
                        st.session_state.chat_history.append({"role": "model", "parts": [{"text": stream_text if stream_text else "(see above)"}]})
                    else:
                        try:
                            model_name = 'models/gemini-1.5-flash-latest'
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content(
                                content,
                                generation_config=genai.GenerationConfig(
                                    temperature=temperature,
                                    max_output_tokens=max_output_tokens,
                                ),
                            )
                            if response.prompt_feedback and response.prompt_feedback.safety_ratings and any(rating.blocked for rating in response.prompt_feedback.safety_ratings):
                                st.warning("Content generation was blocked due to safety concerns. Please try a different prompt or image.")
                            elif response.candidates and response.candidates[0].finish_reason == 2:
                                st.warning("Content generation stopped prematurely due to safety concerns. Please try a different prompt or image.")
                            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                                if image:
                                    st.session_state.chat_history.append({"role": "user", "parts": [{"mime_type": "image/png", "data": img_byte_arr}, {"text": user_input}]})
                                elif user_input:
                                    st.session_state.chat_history.append({"role": "user", "parts": [{"text": user_input}]})
                                st.session_state.chat_history.append({"role": "model", "parts": response.candidates[0].content.parts})
                                output_text = "".join([p.text for p in response.candidates[0].content.parts if hasattr(p, 'text')])
                                container = st.empty()
                                stream_text = ""
                                for word in output_text.split():
                                    stream_text += word + " "
                                    container.markdown(stream_text)
                                    time.sleep(0.04)
                            else:
                                st.warning("Content generation failed to return a valid response.")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

    with colB:
        st.markdown("**Resource Usage**")
        tokens = estimate_token_usage(user_input)
        st.json({"Estimated tokens": tokens, "Estimated cost (USD)": f"${tokens*0.0001:.4f}"})
        st.markdown("---")
        st.markdown("**Chat History**")
        for message in st.session_state.chat_history[::-1]:
            role = "user" if message["role"] == "user" else "assistant"
            with st.chat_message(role):
                for part in message["parts"]:
                    if isinstance(part, dict) and "text" in part:
                        st.markdown(part["text"])
                    elif hasattr(part, 'text'):
                        st.markdown(part.text)

if __name__ == "__main__":
    main()
