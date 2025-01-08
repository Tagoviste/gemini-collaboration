import streamlit as st
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel


# Update project_id
project_id = "inbound-descent-426711-g1"

vertexai.init(project=project_id, location="us-central1")
print("Vertex AI SDK initialized.")
print(f"Vertex AI SDK version = {vertexai.__version__}")

model = GenerativeModel("gemini-1.5-pro-002")

# Generation Config
config = GenerationConfig(
    max_output_tokens=8192, temperature=0.7, top_p=1, top_k=32
    )


def analyze_audio(audio_file):
    """
    Analyzes the audio conversation using Vertex AI and returns the results.

    Args:
      audio_file: The audio file to analyze.

    Returns:
      A dictionary containing the analysis results.
    """
    try:

        # Construct the prompt
        prompt = f"""
        You are a conversation coach.
        Please analyze the following audio conversation and provide feedback on the tone, kindness, and interruptions:

        Focus on these aspects:

        * Determine what happens in the audio
        * Understand the hidden meaning of the audio
        * If there are dialogues, identify the talking personas
        * Make sure the description is clear and helpful
        * Provide a helpful feedback to assist in improving


        <audio:{audio_file}>
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

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Analyze the audio
    analysis_results = analyze_audio(uploaded_file)

    # Display the results
    if "analysis" in analysis_results:
        st.subheader("Conversation Analysis:")
        st.write(analysis_results["analysis"])
    elif "error" in analysis_results:
        st.error(f"An error occurred: {analysis_results['error']}")