import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from config import *

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(INDEX_NAME)


def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def ingest():
    for file in os.listdir("data"):
        war = file.replace(".txt", "")

        with open(f"data/{file}", "r") as f:
            text = f.read()

        for i, chunk in enumerate(chunk_text(text)):
            embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            ).data[0].embedding

            index.upsert([
                {
                    "id": f"{war}-{i}",
                    "values": embedding,
                    "metadata": {
                        "war": war,
                        "text": chunk
                    }
                }
            ])

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()