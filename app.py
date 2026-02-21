import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from config import *

# Page config
st.set_page_config(page_title="World Wars AI", page_icon="🌍")

st.title("🌍 World Wars Intelligence Assistant")
st.markdown("---")

selected_war = st.selectbox(
    "Select War Scope:",
    ["All", "ww1", "ww2", "cold_war"]
)
st.write("Ask anything about WW1, WW2, or the Cold War.")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def ask_question(question):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    if selected_war == "All":
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
    else:
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={"war": selected_war}
        )
    

    context = "\n\n".join(
        [match["metadata"]["text"] for match in results["matches"]]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a military historian. Answer using only the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )

    return response.choices[0].message.content, results


# Input box always visible
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer, results = ask_question(question)

        st.subheader("📜 Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Context"):
            for i, match in enumerate(results["matches"]):
                st.markdown(f"**Chunk {i+1} | Score: {match['score']:.4f}**")
                st.write(match["metadata"]["text"])