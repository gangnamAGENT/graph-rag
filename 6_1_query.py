import os
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

chain = GraphCypherQAChain.from_llm(
    llm = init_chat_model(model=OPENAI_MODEL),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

result = chain.invoke({
    "query": "손해배상금이 언급된 부분 찾아줘"
})

print(result)