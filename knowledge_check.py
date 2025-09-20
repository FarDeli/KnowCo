import os
from openai import OpenAI
from elevenlabs.client import ElevenLabs
import sounddevice as sd
import numpy as np
import tempfile
from scipy.io.wavfile import write
from utils.key import elevenlab_key, openai_key
import vertexai
from typing import List, Union

from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel, Part as PreviewPart
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from vertexai.language_models import TextEmbeddingModel
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
from utils.helper_functions import (process_pdf_folder,
                                    process_pdfs,
                                    clean_excerpt,
                                    openai_token_count,
                                    chunk_document,
                                    embed_chunks,
                                    image_description_prompt,
                                    answer_query_prompt,
                                    prompts_call,
                                    retrieve_top_k,
                                    embedding_functions)
from utils.key import (json_path,
                       project_id,
                       location,
                       openai_key,
                       path_pdf,
                       path_image)


# --- Initialize clients ---
client_openai = OpenAI(api_key=openai_key)
client_eleven = ElevenLabs(api_key=elevenlab_key)

# --- Voices (replace with your preferred ElevenLabs voices) ---
COACH_VOICE = "EXAVITQu4vr4xnSDxMaL"   # calm/professional coach voice
TRAINEE_VOICE = "MF3mGyEYCl7XYWbV9V6O" # trainee/agent voice

# --- Functions ---

import random


def generate_coach_question(multimodal_model, embedding_model, chunk_cluster, vectors):
    """
    Use RAG context to generate a training question from company documents.
    """
    # Instead of random sample, ask: "give me a question to train trainee"
    retrieval_query = "Generate a training question about company knowledge."
    top_chunks = retrieve_top_k(retrieval_query, embedding_model, chunk_cluster, vectors, k=3)

    context = "\n".join([chunk for chunk, _ in top_chunks])

    prompt = f"""
    You are a professional coach. 
    Based on the following company document context, generate ONE short, clear question
    that tests the trainee’s knowledge. Do not provide the answer, only the question.

    Context:
    {context}
    """

    response = multimodal_model.generate_content(prompt)
    return response.text.strip()


def generate_coach_line(context: str) -> str:
    """Generate AI coach dialogue using OpenAI"""
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional coach training a new employee. "
                    "Your role is to ask clear, constructive questions about the company "
                    "and its documents to test the trainee’s knowledge. "
                    "Keep questions short, supportive, and focused. "
                    "Do not answer the questions yourself."
                )
            },
            {"role": "assistant", "content": context}
        ]
    )
    return response.choices[0].message.content


def speak_with_elevenlabs(text: str, voice_id: str, filename: str = "output.mp3"):
    """Convert text to voice and play (macOS uses afplay)"""
    audio = client_eleven.text_to_speech.convert(
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        text=text,
        output_format="mp3_44100_128"
    )
    with open(filename, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)
    os.system(f"afplay {filename}")  # macOS playback


def listen_to_microphone() -> str:
    """Capture voice from mic and transcribe with OpenAI"""
    duration = 5   # seconds to record
    samplerate = 16000
    print("🎙️ Speak now (trainee)...")

    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        write(f.name, samplerate, recording)
        temp_path = f.name

    with open(temp_path, "rb") as f:
        transcript = client_openai.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )
    text = transcript.text.strip()
    print(f"Trainee (you): {text}")
    return text


# --- Coaching Session Loop ---
def main():
    context = "Start the session by asking the trainee a simple question about company procedures."
    print("🎓 Coaching session started. Say 'quit' to stop.\n")

    multimodal_model_2_0_flash = GenerativeModel("gemini-2.0-flash-001")
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    # Process the PDF once
    process_pdf = process_pdf_folder(path_pdf, path_image, "image_description_prompt", multimodal_model_2_0_flash)
    full_text = "\n".join(
        [
            page_data.get("text", "")
            + "\n".join(page_data.get("tables", []))
            + "\n".join(page_data.get("images", []))
            for pdf_data in process_pdf.values()
            for page_data in pdf_data
        ]
    )
    full_text_clean = clean_excerpt(full_text)

    # Create chunks + embeddings once
    chunk_cluster = chunk_document(
        full_text_clean,
        method="cluster",
        embedding_function=embedding_functions
    )
    safe_texts, vectors, failures = embed_chunks(chunk_cluster, embedding_model)

    # Coaching loop
    while True:
        # Step 1: AI Coach generates a question grounded in documents
        coach_question = generate_coach_question(
            multimodal_model_2_0_flash,
            embedding_model,
            chunk_cluster,
            vectors
        )
        print(f"\n📢 Coach (AI): {coach_question}")
        speak_with_elevenlabs(coach_question, COACH_VOICE, "coach.mp3")

        # Step 2: Trainee answers by voice or special command
        trainee_answer = listen_to_microphone()

        # Allow quit/help via voice OR typed fallback
        if trainee_answer.lower().strip() in ["quit", "exit", "stop"]:
            print("👋 Goodbye!")
            break

        if trainee_answer.lower().strip() == "help":
            # Retrieve context + answer from documents
            top_chunks = retrieve_top_k(coach_question, embedding_model, chunk_cluster, vectors, k=3)
            context = "\n".join([chunk for chunk, _ in top_chunks])

            prompt = f"""
            Answer the following question based strictly on the provided context.

            Context:
            {context}

            Question: {coach_question}
            """
            response = multimodal_model_2_0_flash.generate_content(prompt)

            print("\n--- Document Assistant Answer ---")
            print(response.text)
            speak_with_elevenlabs(response.text, COACH_VOICE, "assistant.mp3")
        else:
            # Replay trainee’s own answer in trainee voice
            speak_with_elevenlabs(trainee_answer, TRAINEE_VOICE, "trainee.mp3")
            print("✅ Answer noted! (Say 'help' if you want to check the doc.)")


if __name__ == "__main__":
    main()
