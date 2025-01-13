import streamlit as st
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# Update project_id
project_id = "<project_id>"

vertexai.init(project=project_id, location="us-central1")
print("Vertex AI SDK initialized.")
print(f"Vertex AI SDK version = {vertexai.__version__}")

model = GenerativeModel("gemini-1.5-pro-002")

# Generation Config
config = GenerationConfig(
    max_output_tokens=8192, temperature=0.7, top_p=1, top_k=32
)

def analyze_audio(prompt, uploaded_file):
    """
    Analyzes the audio conversation using Vertex AI and returns the results.

    Args:
      audio_file: The audio file to analyze.
      prompt: The prompt to use for the analysis.

    Returns:
      A dictionary containing the analysis results.
    """
    try:
        # Construct the prompt with the audio file
        prompt = f"""
        {prompt} 

        <audio:{uploaded_file}>
        """
        
        # Generate text using the model
        response = model.generate_content(prompt, generation_config=config)
        analysis = response.text
        print(f"Analysis: {analysis}")
        return {"analysis": analysis}

    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return {"error": str(e)}

st.title("Conversation Coach")

# Text input for the prompt
prompt = st.text_area("Enter your prompt here:", value=f"""


""")

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None and prompt:
    # Analyze the audio
    analysis_results = analyze_audio(prompt, uploaded_file)

    # Display the results
    if "analysis" in analysis_results:
        st.subheader("Conversation Analysis:")
        st.write(analysis_results["analysis"])
    elif "error" in analysis_results:
        st.error(f"An error occurred: {analysis_results['error']}")