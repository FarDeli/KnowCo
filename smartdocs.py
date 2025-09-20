import os
import numpy as np
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
                                    retrieve_top_k,
                                    embedding_functions)
from utils.key import (json_path,
                       project_id,
                       location,
                       openai_key,
                       path_pdf,
                       path_image)


def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path
    vertexai.init(project=project_id, location=location)

    print("Welcome to KnowCo! We’re excited to have you on board. Here, "
          "you’ll find a supportive environment where you can focus, grow, and thrive.")

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

    # Enter question/answer loop
    while True:
        ask_query = input("\nAsk a question related to the document (or type 'quit' to exit): ").strip()
        if ask_query.lower() == "quit":
            print("Goodbye!")
            break

        top_chunks = retrieve_top_k(ask_query, embedding_model, chunk_cluster, vectors, k=3)
        context = "\n".join([chunk for chunk, _ in top_chunks])

        prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {ask_query}"
        response = multimodal_model_2_0_flash.generate_content(prompt)

        print("\n--- Answer ---")
        print(response.text)


if __name__ == "__main__":
    main()
