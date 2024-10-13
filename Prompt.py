from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser   
from langchain_core.messages import HumanMessage,AIMessage
from pinecone import Pinecone,ServerlessSpec
import pinecone
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
import pinecone
from google.cloud import aiplatform
from dotenv import load_dotenv
import os


load_dotenv()
pinecone=Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
aiplatform.init(project=os.getenv("GCP_KEY"))



llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro")

#prompt
sys_template='''
You are an AI assistant designed to provide fast and accurate customer support by using past ticket data and product information. Follow these steps:

Primary Search (Past Ticket Information):

Upon receiving a query, first search the vector database of {past_info} (past ticket data) for any relevant information or previous similar responses.
If relevant information is found in past responses, generate a concise and helpful response based on this data.
Fallback Search (Product Information):

If no relevant information is found in the past ticket data, or the query requires more specific product details, search the vector database of {products}.
Retrieve product-specific details and generate a response based on this information.
Response Generation:

Whether the information comes from past tickets or product data, provide a clear, accurate, and user-friendly response.
Prioritize responses that fully resolve the user's query, combining both sources of information if needed.

User-Centric Response:
Ensure responses are customer-friendly, providing helpful and easy-to-understand solutions.
'''

Prompt=ChatPromptTemplate.from_messages(
    [
        ("system",sys_template),
        ("user","{query}")
    ]
)

out=StrOutputParser()
chatbot_chain= Prompt | llm | out




#report generation

report_template='''
You are an AI assistant tasked with generating a structured summary of customer support sessions. 
Based on the provided session, extract key information and produce a report in the following JSON format:
    [
  "product": "<product name: string>",
  "issues": "<list of issues: array of strings (keywords)>",
  "solution_offered": "<summary of response>"
    ]
Follow these steps:

Extract Product Information:

Identify the product name mentioned in the session. Use concise, accurate product names.
If no product is explicitly mentioned, leave the product field empty or use the general category.
Identify Issues:

Analyze the query to identify and extract key issues or problems raised by the user.
Represent these issues as an array of keywords, summarizing each problem in a few words.
Summarize the Solution Offered:

Based on the response provided in the session, generate a brief summary of the solution offered to resolve the issue.
[
  "product": "<extracted product name>",
  "issues": ["<issue 1>", "<issue 2>", "..."],
  "solution_offered": "<brief summary of response>"
]


'''

ReportPrompt=ChatPromptTemplate.from_messages(
    [
        ("system",report_template),
        ("user","{session_data}")
    ]
)

report_chain= ReportPrompt| llm |out

index_name = "input-index2"
try:
    found=False
    for i in pinecone.list_indexes():
        if i["name"]==index_name:
            found=True
    if (found==False):
        pinecone.create_index(index_name, dimension=768,spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) )
except():
    pass
index = pinecone.Index(index_name)
print(pinecone.list_indexes())

def generate_embedding(text):
    client = aiplatform.gapic.PredictionServiceClient()
    endpoint = client.endpoint_path(
        project=os.getenv("GEMINI_API_KEY"),
        location="us-central1",
        endpoint="gemini-1.5-pro"
    )
    response = client.predict(
        endpoint=endpoint,
        instances=[{"content": text}],
        parameters={"embedding": True}
    )
    return response.predictions[0]["embedding"]





def upsert_embedding(query):
    embedding = generate_embedding({query})
    data = {
        "id": str(uuid.uuid4()),
        "values": embedding
    }
    index.upsert([(data["id"], data["values"])])

