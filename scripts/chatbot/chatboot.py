from groq import Groq
from dotenv import load_dotenv  # Import the load_dotenv function
import PyPDF2
import os

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

async def extract_pdf_data(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_path = r"D:\mojahed_chatbot\data\Palestine.pdf"
pdf_text = extract_pdf_data(pdf_path) 


client = Groq(api_key=api_key)

query = input("How can I help u today😊?")


completion = client.chat.completions.create(
    model="deepseek-r1-distill-llama-70b",
    messages=[
        {
            "role": "user",
            "content": f"{query}"
        },
        
    ],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=0.95,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")

print(50*"=")
print(50*"=")