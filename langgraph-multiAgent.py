import utils
import os
from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Definir el modelo LLM
os.environ["OPENAI_API_KEY"] = utils.config["EMB"]["OPENAI_API_KEY"]
model = ChatOpenAI(model="gpt-4")

members = ["jira_agent", "confluence_agent", "test_case_agent"]
TESTER_PROMPT = """Sos un experto generando casos de prueba en base a requerimientos de software. Cuando se te solicita generar casos de prueba sobre un proyecto o sprint, \
     realizas las siguientes tareas: \
     1. consultas en JIRA cuales son los tickets del proyecto o sprint que indique el usuario que no tienen casos de test generados. \
     2. buscas en Confluence cual es la definición de cada uno de esos tickets. \
     3. teniendo la definición, escribís los casos de test de cada uno. \
     4. con el resultado de los casos de test, cargas esos casos como tickets en JIRA para que se registren y posteriormente se ejecuten. \
     Para llevar adelante estas tareas, dispones de las siguientes herramientas: \
     1. jira_agent: Permite hacer consultas sobre JIRA y enviar el resultado de los casos de test al final para que se carguen como tickets linkeados a los requerimientos. \
     2. confluence_agent: Permite hacer consultas sobre CONFLUENCE para obtener la descripción funcional de cada requerimiento. \
     3. test_case_agent: Es un experto en escribir casos de test en base a descripciones funcionales de requerimientos. \
     Cada una de estas herramientas va a realizar lo que le pidas y va a devolver el resultado y el estado. \
     Ante cada interaccion, tenes que responder con alguna de las herramientas para continuar o si necesitas consultar algo al usuario responde con __humano__. Cuando finalices, responde con __end__."""
options = ["__humano__","__end__"] + members

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", TESTER_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Dada la conversacion arriba, quien debería actuar a continuacion?"
            " O debería finalizar? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

class routeResponse(BaseModel):
    next: Literal[*options]

# Definir el estado del agente
class AgentState(MessagesState):
    next: Literal["jira_agent", "confluence_agent", "test_case_agent", "__end__"]


# Definir el Supervisor
def supervisor(state: AgentState):
    supervisor_chain = prompt | model.with_structured_output(routeResponse)
    return supervisor_chain.invoke(state)


# Definir el Agente de JIRA
def jira_agent(state: AgentState):
    # Lógica para interactuar con JIRA
    response = model.invoke("Procesando solicitud en JIRA...")
    return {"messages": [response]}


# Definir el Agente de Confluence
def confluence_agent(state: AgentState):
    # Lógica para interactuar con Confluence
    response = model.invoke("Procesando solicitud en Confluence...")
    return {"messages": [response]}


# Definir el Agente Generador de Casos de Prueba
def test_case_agent(state: AgentState):
    # Lógica para generar casos de prueba
    response = model.invoke("Generando casos de prueba...")
    return {"messages": [response]}


# Construir el Gráfico de Estados
builder = StateGraph(AgentState)
builder.add_node(supervisor)
builder.add_node(jira_agent)
builder.add_node(confluence_agent)
builder.add_node(test_case_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", lambda state: state["next"])
builder.add_edge("jira_agent", "supervisor")
builder.add_edge("confluence_agent", "supervisor")
builder.add_edge("test_case_agent", "supervisor")

# Compilar el gráfico
orchestrator_agent = builder.compile()


# Función para interactuar con el Agente Orquestador
def orchestrator_interaction(user_input):

    initial_state = AgentState(messages=[{"role": "user", "content": user_input}])
    for s in orchestrator_agent.stream(initial_state ):
        if "__end__" not in s:
            print(s)
            print("----")
    return s


# Ejemplo de uso
user_input = "Genera los casos de prueba para el proyecto CUX"
print(orchestrator_interaction(user_input))
