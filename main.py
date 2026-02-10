import time
import json
import urllib.parse
import requests
import re
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from vector import setup_vector_db, retrieve_similar_docs

# ==== Setup vector DB and model ====
matrix_quotes = setup_vector_db()

model = OllamaLLM(model="hf.co/jurieyel/Llama3-sqlcoder-8b-4bit-GGUF-q4_K_M:latest")

# ==== Prompt Template ====
template = """
You are an expert in answering SQL-related questions.

You will be provided with two relevant examples. Each example includes:
- A SQL task description
- The corresponding SQL query
- An explanation of the SQL statement
- The schema or context used

Using these examples and your own SQL knowledge, generate an appropriate SQL query to answer the given question based on the provided schema.

***IMPORTANT:*** Output only the SQL query.  
Do NOT include any explanations, markdown code fences, or other text.

Relevant Example 1:
{reviews1}

Relevant Example 2:
{reviews2}

User Question:
{question}

Schema:
{schema}
"""

prompt = PromptTemplate.from_template(template)
chain = prompt | model

# ==== Upstash Configuration ====
UPSTASH_URL = "https://magnetic-ringtail-19701.upstash.io"
UPSTASH_TOKEN = "AUz1AAIjcDE4ZGEyYzNkYzNhNmU0ZjI1OTc5NjljZWJiY2EwZmU4Y3AxMA"

def upstash_lpop(key):
    return requests.post(
        f"{UPSTASH_URL}/LPOP/{key}",
        headers={"Authorization": UPSTASH_TOKEN}
    ).json().get("result")

def upstash_set(key, value):
    return requests.post(
        f"{UPSTASH_URL}/SET/{key}/{urllib.parse.quote(value)}",
        headers={"Authorization": UPSTASH_TOKEN}
    ).json()

# ==== SQL Generator ====
def extract_sql(text: str) -> str:
    # First try to grab ``````
    m = re.search(r"``````", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: grab any triple-backtick block
    m = re.search(r"``````", text)
    if m:
        return m.group(0).strip("` \n")
    # Otherwise return the whole string
    return text.strip()
def generate_query(question, schema):
    reviews = retrieve_similar_docs(matrix_quotes, question, schema)
    metadata1 = reviews['metadatas'][0][0]
    metadata2 = reviews['metadatas'][0][1]

    reviews1 = f"""Task: {metadata1['prompt']}
SQL: {metadata1['sql']}
Explanation: {metadata1['explanation']}
Context: {metadata1['context']}"""

    reviews2 = f"""Task: {metadata2['prompt']}
SQL: {metadata2['sql']}
Explanation: {metadata2['explanation']}
Context: {metadata2['context']}"""

    result = chain.invoke({
        "reviews1": reviews1,
        "reviews2": reviews2,
        "question": question,
        "schema": schema
    })
    result = result.strip()
    return result.split(";")[0]

while True:
    try:
        request_data = upstash_lpop("sql_requests")
        if not request_data:
            time.sleep(2)
            continue

        request_data = urllib.parse.unquote(request_data)
        request = json.loads(request_data)
        question = request.get("question")
        schema = request.get("schema")
        request_id = request.get("id")

        if not question or not schema or not request_id:
            print("⚠️ Invalid request format")
            continue

        try:
            formatted_sql = generate_query(question, schema)
            upstash_set(f"result:{request_id}", json.dumps({"sql": formatted_sql}))
            print(f"✅ SQL generated for {request_id}")
        except Exception as e:
            print(f"❌ Error generating SQL: {e}")
            upstash_set(f"result:{request_id}", json.dumps({"error": str(e)}))

    except Exception as e:
        print(f"❌ Worker error: {e}")
        time.sleep(1)
    time.sleep(1)
