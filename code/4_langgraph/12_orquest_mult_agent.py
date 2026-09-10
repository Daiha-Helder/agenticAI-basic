from IPython.display import Image, display, Markdown
import sqlite3
from typing import TypedDict, List
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from tavily import TavilyClient


import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_APY_KEY = os.getenv('GEMINI_APY_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

conn = sqlite3.connect('checkpoints.db', check_same_thread=False)
memory = SqliteSaver(conn)

model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    temperature=0
)

PLAN_PROMPT = """
Você é um escritor especialista com a tarefa de criar um esboço de alto nível para uma redação. \
Escreva esse esboço para o tópico fornecido pelo usuário. Apresente um plano da redação junto com quaisquer notas ou instruções relevantes para as seções.
"""

WRITER_PROMPT = """
Você é um assistente de redação com a tarefa de escrever excelentes redações de 5 parágrafos. \
Gere a melhor redação possível para a solicitação do usuário e o esboço inicial. \
Se o usuário fornecer críticas, responda com uma versão revisada das suas tentativas anteriores. \
Utilize todas as informações abaixo conforme necessário:

------

{content}
"""

REFLECTION_PROMPT = """
Você é um professor corrigindo uma redação submetida. \
Gere uma crítica e recomendações para a submissão do usuário. \
Forneça recomendações detalhadas, incluindo pedidos sobre extensão, profundidade, estilo, etc.
"""

RESEARCH_PLAN_PROMPT = """
Você é um pesquisador encarregado de fornecer informações que podem ser usadas ao escrever a sequinte redação. \
Gere uma lista de consultas de pesquisa que recolham quaisquer informações relevantes. Gere no máximo 3 consultas.
"""

RESEARCH_CRITIQUE_PROMPT = """
Você é um pesquisador encarregado de fornecer informações que podem ser usadas ao fazer quaisquer revisões solicitadas (conforme descrito abaixo). \
Gere uma lista de consultas de pesquisa que recolham quaisquer informações relevantes. Gere no máximo 3 consultas.
"""

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]

def plan_node(state: AgentState):

    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(state['task'])
    ]

    response = model.invoke(messages)
    return {"plan":response.content}

def research_plan_node(state: AgentState):

    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])

    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(q, max_results=2)
        for r in response['results']:
            content.append(r['content'])

    return {"content": content}

