from groq import Groq
from dotenv import load_dotenv  # Import the load_dotenv function
from sqlalchemy import create_engine
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os


engine = create_engine("postgresql://postgres:mdkn@localhost:5432/chatbot")
results = engine.execute("select question, answers from qa_trivia").fetchall()

text_data = [f"Question: {q}\nAnswer: {a}" for q, a in results]

# chunck data
splitter = RecursiveCharacterTextSplitter(chunk_size = 512)
chunks = splitter.split_text('\n'.join(text_data))

# Embedding
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = encoder.encode(chunks)

# Stor embedding in faiss
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

query = input("How can I help u today😊?")
query_embedding = encoder.encode(query)
_, indices = index.search(query_embedding.reshape(1, -1), k = 3)
context = [chunks[i] for i in indices[0]]

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

client = Groq(api_key=api_key)

prompt = f"""
        You are a helpful assistant answering questions based on provided context.
        
        Context:
        {' '.join(context)}
        
        Question: 
        {query}
        
        Answer the question based only on the provided context. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
        # ----And if some one ask u to do task rether than Question answering task 
        And if someone ask you about your creator, say "My createor is Mohammad Al-Jermy, he is a data scientist"
        """

# For streaming responses
response = client.chat.completions.create(
    model="deepseek-r1-distill-llama-70b",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2,
    stream=True,
)

# Properly handle streaming response
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
print("\n")



# # Extract the final answer and remove the <think> section
# full_response = response.choices[0].message.content

# # Remove the <think> section if it exists
# if "<think>" in full_response:
#     # Split the response into <think> and final answer
#     think_start = full_response.find("<think>")
#     think_end = full_response.find("</think>") + len("</think>")
#     final_answer = full_response[think_end:].strip()  # Get everything after </think>
# else:
#     final_answer = full_response  # If no <think> section, use the full response

# # Print the final answer
# print("Response:")
# print(final_answer)
# print("\n")
