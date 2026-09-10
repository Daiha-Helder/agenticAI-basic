from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, BaseMessage, AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

from IPython.display import Image, display
from typing_extensions import TypedDict
import sqlite3
from datetime import date
from typing import Annotated
from uuid import uuid4

import os 
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

def reduce_messages(
        left: list[AnyMessage],
        right: list[AnyMessage]
    ) -> list[AnyMessage]:

    for message in right:
        if not message.id:
            message.id = str(uuid4())

    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            merged.append(message)
    return merged


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


tool = TavilySearch(max_results=2)

class Agent: 

    def __init__(
            self,
            model,
            tools,
            system="",
            checkpointer = None
            ):

        self.system = system

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)

        graph.set_entry_point("llm")
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
            )
        graph.add_edge("action", "llm")

        self.graph = graph.compile(
            checkpointer = checkpointer,
            interrupt_before = ["action"] # Adiciona interrupção antes de chamar a ação
            )

        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_gemini(
            self, 
            state: AgentState):

        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        print("Mensagens enviadas ao modelo:", messages)
        messages = self.model.invoke(messages)
        return {'messages': [messages]}

    def exists_action(
            self,
            state: AgentState
            ):

        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(
            self,
            state: AgentState
            ):

        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:

            print(f"Calling Tool: {t['name']} with args: {t['args']}")

            result = self.tools[t['name']].invoke(t['args'])
            results.append(
                ToolMessage(
                    tool_call_id=t['id'],
                    name=t['name'],
                    content=str(result)
                )
            )

        print("Returning to LLM after action!")
        return {'messages': results}


current_date = date.today().strftime("%d/%m/%Y") 

prompt = f"""Você é um assistente de pesquisa inteligente e altamente atualizado. \
Sua principal prioridade é encontrar as informações mais RECENTES e em TEMPO REAL sempre que possível. \
A data atual é {current_date}. \
Ao buscar sobre o tempo ou eventos que se referem a "hoje" ou "agora", \
você DEVE **incluir a data atual '{current_date}' na sua consulta para a ferramenta de busca**. \
Por exemplo, se a pergunta é "tempo em cidade x hoje", a consulta para a ferramenta deve ser "tempo em cidade x {current_date}". \
Ignore ou descarte informações que claramente se refiram a datas passadas ou futuras ao responder perguntas sobre "hoje". \
Use o mecanismo de busca para procurar informações, sempre buscando o 'hoje' ou o 'agora' quando o contexto indicar. \
Você tem permissão para fazer múltiplas chamadas (seja em conjunto ou em sequência). \
Procure informações apenas quando tiver certeza do que você quer. \
Se precisar pesquisar alguma informação antes de fazer uma pergunta de acompanhamento, você tem permissão para fazer isso!
"""

model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    temperature = 0
)

abot = Agent(
    model, 
    [tool], 
    system=prompt, 
    checkpointer=memory
)

session_id = str(uuid4())
print(f"DEBUG: Iniciando nova conversa com ID: {session_id}\n")

user_message = "Como está o tempo em São Paulo hoje?"
messages = [HumanMessage(content=user_message)]
thread_config = {"configurable": {"thread_id": session_id}}

print("--- Etapa 1: Agente processa a entrada e decide a ação ---")
print(f"Você: {user_message}")

for event in abot.graph.stream({"messages": messages}, thread_config):

    for k,v in event.items():
        if k == "llm":
            last_message = v.get('messages', [])[-1]
            if (isinstance(last_message, AIMessage)) and (last_message.tool_calls):
                print(f"\nAgente (decisão): {last_message.tool_calls}")
                print("\n--- AGENTE PAUSADO: Intervenção Humana Necessária ---")
            else:
                print(f"\nAgente (resposta direta/sem tool_calls): {last_message.content}")
                print("\n--- AGENTE PAUSADO (resposta direta, sem ação pendente) ---")

current_state = abot.graph.get_state(thread_config)
last_state_message = current_state.values['messages'][-1]

if current_state and (current_state.next == ('action', )) and isinstance(last_message, AIMessage) and (last_state_message.tool_calls):
    tool_calls_pending = last_state_message.tool_calls
    if tool_calls_pending:
        print("\nO agente decidiu executar a(s) seguinte(s) ação(ões) de ferramenta:")
        for tc in tool_calls_pending:
            print(f"- Ferramenta: {tc['name']}, Argumentos: {tc['args']}")

        user_input = input("\nVocê deseja que o agente execute esta(s) ação(ões)? (sim/não): ").lower()

        if user_input == 'sim':

            print("\n --- Etapa 2: Retomando a execução (Agente executará a ação) ---")
            for event in abot.graph.stream(None, thread_config):
                for k, v in event.items():
                    if k=='action':
                        print(f"DEBUG: Ferramenta executada e resultado retornado: {v}") 
                    elif k == 'llm':
                        final_response_message = v.get('messages', [])[-1].content
                        print(f"\nAgente (resposta final): {final_response_message}")
                    elif k==END:
                        print(f"DEBUG: Grafo terminou a execução.")
            print("\n --- FIM DA INTERAÇÃO ---")
        else:
            print("\nExecução da ação cancelada pelo usuário.")
            print(" --- FIM DA INTERAÇÃO ---")
    else:
        print("\nO agente não decidiu nenhuma ação de ferramenta apesar da pausa. Interação encerrada.")
else:
    print("\nO agente respondeu diretamente ou não pausou em uma ação. Não há ações pendentes para aprovar.")
    if current_date:
        final_response_message = current_state.values['messages'][-1].content
        print(f"Agente (resposta direta): {final_response_message}")
    print("--- FIM DA INTERAÇÃO ---")
                    


# print("\n--- Tentando Gerar PNG do Grafo via Mermaid ---")
# try:
#     image_data = abot.graph.get_graph().draw_mermaid_png()
#     display(Image(data=image_data))

# except AttributeError:
#     print("Método `.draw_mermaid_png()` não encontrado ou não suportado.")
#     print("Tentando gerar apenas o código Mermaid...")
#     try:
#         mermaid_code = abot.graph.get_graph().draw_mermaid()
#         print("\n--- Código Mermaid Gerado (Cole em https://mermaid.live/) ---")
#         print(mermaid_code)
#     except Exception as e_mermaid:
#         print(f"Erro ao gerar código Mermaid: {e_mermaid}")

# except Exception as e:
#     print(f"Erro inesperado ao tentar gerar o grafo: {e}")

# try:
#     mermaid_code = abot.graph.get_graph().draw_mermaid()

#     with open("grafo_agente.mmd", "w", encoding="utf-8") as arquivo:
#         arquivo.write(mermaid_code)

#     print("Código Mermaid salvo em: grafo_agente.mmd")
#     print("Você pode visualizá-lo em: https://mermaid.live/")

# except Exception as e:
#     print(f"Erro ao gerar o código Mermaid: {e}")