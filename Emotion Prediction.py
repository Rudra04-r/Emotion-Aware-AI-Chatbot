
!pip install -q torch transformers gradio matplotlib


import torch
from transformers import pipeline
import gradio as gr
import matplotlib.pyplot as plt
from collections import Counter

print("Loading Emotion Model...")


device = 0 if torch.cuda.is_available() else -1


emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device
)

print("Model loaded successfully!")


emotion_history = []


def generate_response(message, history):

    if not message or not message.strip():
        return "Please type something!"

    text = message.lower()

    
    if any(word in text for word in ["stress", "stressed", "anxious", "overwhelmed"]):
        emotion = "stress"
        score = 1.0
    else:
        result = emotion_classifier(message)
        emotion = result[0]['label']
        score = result[0]['score']

    
    emotion_history.append(emotion)


    if emotion in ["sadness", "stress", "fear"]:
        reply = "I'm really sorry you're feeling this way. It sounds tough, but you're not alone. Want to talk more about it?"

    elif emotion == "joy":
        reply = "That's wonderful to hear! 😊 What made you feel so happy?"

    elif emotion == "anger":
        reply = "I understand you're upset. Try taking a deep breath. Do you want to share what happened?"

    elif emotion == "disgust":
        reply = "That sounds unpleasant. I'm sorry you had to experience that."

    elif emotion == "surprise":
        reply = "That’s surprising! 😮 Tell me more about it!"

    else:
        reply = "I'm here to listen 🙂 Tell me more."

    return f"🧠 Emotion: {emotion.capitalize()} ({score:.2f})\n💬 {reply}"


def show_emotion_chart():
    if len(emotion_history) == 0:
        return None

    count = Counter(emotion_history)

    plt.figure()
    plt.bar(count.keys(), count.values())
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")

    return plt


with gr.Blocks() as app:

    gr.Markdown("# 🤖 Emotion-Aware AI Chatbot")
    gr.Markdown("Chat with AI and see emotion analytics!")

    chat = gr.ChatInterface(fn=generate_response)

    gr.Markdown("## 📊 Emotion Analytics Dashboard")
    btn = gr.Button("Show Emotion Chart")
    plot = gr.Plot()

    btn.click(show_emotion_chart, outputs=plot)

app.launch(share=True)