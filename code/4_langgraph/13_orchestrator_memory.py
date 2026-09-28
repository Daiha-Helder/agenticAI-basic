import google.generativeai as genai
from prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.types import Command
from typing import Literal
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

    classification: Literal["ignore", "respond", "notify"] = Field(
        description="A classificação de um e-mail: 'ignore' para e-mails irrelevantes, "
        "'notify' para informações importantes que não precisam de resposta, "
        "'respond' para e-mails que precisam de uma resposta",
    )

llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    temperature=0)

llm_router = llm.with_structured_output(Router)

system_prompt = triage_system_prompt.format(
    full_name = profile["full_name"],
    name = profile["name"],
    examples = None,
    user_profile_background = profile["user_profile_background"],
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

# result = llm_router.invoke([
#     {"role": "system", "content": system_prompt},
#     {"role": "user", "content": user_prompt},
# ])

# %% Testando a `llm_router`

# print("Exemplo teste para verificar funcionamento do `llm_router`")
# print(result)


# %% Construindo as ferramentas

@tool
def write_email(
    to: str, 
    subject: str, 
    content: str
    ) -> str:

    """Escreve e envia um e-mail."""
    # Resposta de placeholder - em um aplicativo real, enviaria o e-mail
    return f"E-mail enviado para {to} com o assunto '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int, 
    preferred_day: str
    )-> str:
    """Agenda uma reunião no calendário."""

    return f"Reunião '{subject}' agendada para {preferred_day} com {len(attendees)} participantes"

@tool
def check_calendar_availability(
    day: str
    ) -> str:

    """Verifica a disponibilidade do calendário para uma determinado dia."""
    return f"Horários disponíveis em {day}: 9:00 AM, 2:00 PM, 4:00 PM"


tools = [write_email, schedule_meeting, check_calendar_availability]

# %% Criação do agente

def create_prompt(state):
    return [
        {
            "role": "system",
            "content": agent_system_prompt.format(
                instructions = prompt_instructions["agent_instructions"],
                **profile,
            ),
        }
    ] + state["messages"]


agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=create_prompt
)

# %% Testando os agentes

# response = agent.invoke(
#     {"messages": [{
#         "role": "user",
#         "content": "qual é minha disponibilidade para terça-feira?"
#     }]}
# )

# response["messages"][-1].pretty_print()

# %% Construindo sistema de agentes 

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

def triage_router(state: State) -> Command[
    Literal["response_agent", "__end__"]
    ]:

    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    
    system_prompt = triage_system_prompt.format(
        full_name = profile["full_name"],
        name = profile["name"],
        user_profile_background = profile["user_profile_background"],
        triage_no = prompt_instructions["triage_rules"]["ignore"],
        triage_notify = prompt_instructions["triage_rules"]["notify"],
        triage_email = prompt_instructions["triage_rules"]["respond"],
        examples = None,
    )

    user_prompt = triage_user_prompt.format(
            author=author, 
            to=to, 
            subject=subject, 
            email_thread=email_thread
        )

    result = llm_router.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
    )

    if result.classification == "respond":
        print("Classificação: RESPONDER - Este e-mail requer uma resposta")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Responda ao e-mail {state['email_input']}",
                }
            ]
        }

    elif result.classification == "ignore":
            print("Classificação: IGNORAR - Este e-mail pode ser ignorado com segurança")
            update = None
            goto = END
    elif result.classification == "notify":
        print("Classificação: NOTIFICAR - Este e-mail contém informações importantes")
        update = None
        goto = END
    else:
        raise ValueError(f"Classificação inválida: {result.classification}")
    return Command(goto=goto, update=update)


email_agent = StateGraph(State)
email_agent = email_agent.add_node("triage_router", triage_router)
email_agent = email_agent.add_node("response_agent", agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile()

# Salvar como PNG
png = email_agent.get_graph(xray=True).draw_mermaid_png()

with open("./data/email_agent.png", "wb") as f:
    f.write(png)

