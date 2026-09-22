import google.generativeai as genai
from prompts import triage_system_prompt, triage_user_prompt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_APY_KEY = os.getenv('GEMINI_APY_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

profile = {
    "name": "Sarah",
    "full_name": "Sarah Chen",
    "user_profile_background": "Engenheira de software sênior liderando uma equipe de 5 desenvolvedores"
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Newsletters de marketing, e-mails de spam, comunicados gerais da empresa",
        "notify": "Membro da equipe doente, notificações do sistema de build, atualizações de status de projeto",
        "respond": "Perguntas diretas de membros da equipe, solicitações de reunião, relatórios de bugs críticos"
    },
    "agent_instructions": "Use estas ferramentas quando apropriado para ajudar a gerenciar as tarefas de Sarah de forma eficiente."
}

email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "Sarah Chen <sarah.chen@company.com>",
    "subject": "Dúvida rápida sobre a documentação da API",
    "body":""""
Olá Sarah,

Eu estava revisando a documentação da API para o novo serviço de autenticação e notei que alguns endpoints parecem estar faltando nas especificações.
Você poderia me ajudar a esclarecer se isso foi intencional ou se devemos atualizar a documentação?

Especificamente, estou procurando por:
- /auth/reflesh
- - /auth/validate

Obrigada!
Alice
"""
}


class Router(BaseModel):
    """Analisa o e-mail não lido e roteia de acordo com seu conteúdo."""

    reasoning: str = Field(
        description="Raciocínio passo a passo por trás da classificação."
    )

    classification = Literal["ignore", "respond", "notify"] = Field(
        description="A classificação de um e-mail: 'ignore' para e-mails irrelevantes, "
        "'notify' para informações importantes que não precisam de resposta, "
        "'respond' para e-mails que precisam de uma resposta",
    )

llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    temperature=0)

llm_router = llm.with_structured_output(Router)

system_prompt = triage_user_prompt.format(
    full_name = profile["full_name"],
    name = profile["name"],
    examples = None,
    user_profile_background = profile["user_profile_ground"],
    triage_no = prompt_instructions["triage_rules"]["ignore"],
    triage_notify = prompt_instructions["triage_rules"]["notify"],
    triage_email = prompt_instructions["triage_rules"]["respond"]
)

user_prompt = triage_user_prompt.format(
    author = email["from"],
    to = email["to"],
    subject = email["subject"],
    email_thread = email["body"]
)
