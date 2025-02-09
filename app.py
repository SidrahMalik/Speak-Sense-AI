import gradio as gr
import whisper
from textblob import TextBlob  

# Load Whisper model
model = whisper.load_model("small")

def transcribe_audio(audio):
    if audio is None:
        return "Please upload an audio file.", "N/A"
    
    # Transcribe the audio
    result = model.transcribe(audio, language="en")
    corrected_text = str(TextBlob(result["text"]).correct())

    # Perform sentiment analysis
    sentiment_score = TextBlob(corrected_text).sentiment.polarity
    
    # Classify sentiment
    if sentiment_score > 0.1:
        sentiment = "🌟 Positive Vibes!"
    elif sentiment_score < -0.1:
        sentiment = "⚠️ Negative Tone Detected"
    else:
        sentiment = "😐 Neutral Statement"

    return corrected_text, sentiment

# Gradio interface
interface = gr.Interface(
    fn=transcribe_audio, 
    inputs=gr.Audio(type="filepath"),  
    outputs=[
        gr.Textbox(label="Speech to Text"), 
        gr.Textbox(label="Tone Analyzing")
    ],  
    title="SpeakSense - AI Transcriber & Tone Analyzer",
    description="Upload an audio file to transcribe speech into text and analyze its tone."
)

# Launch the app
interface.launch(share=True)

