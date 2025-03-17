import os
import utils
from langchain_openai import ChatOpenAI

# Imports para Indexacion
from glob import glob
from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Imports Agente
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict
from typing_extensions import TypedDict, Annotated
import operator


os.environ["LANGCHAIN_TRACING_V2"] = utils.config["LANG"]["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = utils.config["LANG"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "RAG-PMX-Agent-Gradio"

os.environ["OPENAI_API_KEY"] = utils.config["LLM"]["OPENAI_API_KEY"]


class AgentState(TypedDict):
    question: str
    context: List[Document]
    messages: Annotated[List[BaseMessage], operator.add]  # Historial automático
    answer: str


class PMXAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.index_files()
        self.template = """Sos un asistente de soporte sobre un sistema de evaluaciones de performance de empleados. Usa el contexto y el historial de la conversación para responder preguntas.
        El contexto incluye una guia de usuario del sistema y la norma de evaluacion de la empresa.
        Si no sabes la respuesta en base al contexto y a lo que te haya informado el usuario, solo di "no se", no trates de crear una respuesta.
        Mantene la respuesta tan conscisa como sea posible.
        Contexto:
        {context}

        Historial de conversación:
        {chat_history}

        Nueva pregunta:
        {question}

        Respuesta:"""
        self.custom_rag_prompt = PromptTemplate.from_template(self.template)
        # Construir el flujo
        graph_builder = StateGraph(AgentState).add_sequence(
            [self.retrieve, self.generate]
        )
        graph_builder.add_edge(START, "retrieve")

        # Compilar el grafo con soporte automático para mensajes
        memory = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)

    def index_files(self):
        # Indexacion - Load
        # Ruta con un patrón de búsqueda para archivos HTML
        html_files = glob(
            "C:/Users/baiscf/repos_local/Confluence-LangChain/PMX_Manual/tenaris/es/*.html"
        )
        pdf_files = glob(
            "C:/Users/baiscf/repos_local/Confluence-LangChain/PMX_Manual/*.pdf"
        )

        # Cargar y procesar cada archivo
        documents = []
        for file_path in html_files:
            print(f"file path: {file_path}")
            loader = BSHTMLLoader(file_path, open_encoding="utf-8")
            documents.extend(loader.load())

        for file_path in pdf_files:
            print(f"file path: {file_path}")
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                # Agregar metadatos personalizados
                doc.metadata["source"] = file_path  # Ruta del archivo como fuente
                doc.metadata["type"] = "PDF"  # Tipo de documento
                doc.metadata["title"] = (
                    f"PDF: {file_path.split('/')[-1]}"  # Nombre del archivo como título
                )
            documents.extend(loaded_docs)
        # Ahora `documents` contiene todos los documentos cargados.

        # Indexacion - Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(documents)

        # Indexacion - Vector
        os.environ["OPENAI_API_KEY"] = utils.config["EMB"]["OPENAI_API_KEY"]

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        self.vectorstore = InMemoryVectorStore(embeddings)
        # Index chunks
        _ = self.vectorstore.add_documents(documents=all_splits)

    def retrieve(self, state: AgentState):
        """Paso 1: Recuperar documentos relevantes."""
        retrieved_docs = self.vectorstore.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: AgentState):
        """Paso 2: Generar una respuesta basada en el historial."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        # El historial ya está gestionado en `state["messages"]`

        # Formatear los documentos recuperados como una lista amigable
        resources = []
        for doc in state["context"]:
            title = doc.metadata.get("title", "Recurso sin título")
            url = doc.metadata.get("url", "")
            # snippet = doc.page_content[:200]  # Primeras 200 caracteres como resumen
            # resources.append(f"- **{title}**: {snippet}... {f'[Leer más]({url})' if url else ''}")
            resources.append(f"- **{title}**: {f'[Leer más]({url})' if url else ''}")
        # Crear un resumen de los recursos en formato amigable
        resources_summary = (
            "\n".join(resources)
            if resources
            else "No se encontraron recursos relevantes."
        )

        prompt = self.custom_rag_prompt.invoke(
            {
                "question": state["question"],
                "context": docs_content,
                "chat_history": state[
                    "messages"
                ],  # LangGraph mantiene esta lista actualizada
            }
        )
        response = self.llm.invoke(prompt)

        # Combinar la respuesta generada con los recursos recuperados
        friendly_response = (
            f"{response.content}\n\n### Recursos relacionados:\n{resources_summary}"
        )

        # Devolver la respuesta como nuevo mensaje de la IA
        return {
            "answer": friendly_response,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=response.content),
            ],
        }


# config = {"configurable": {"thread_id": "2"}}
# an_agent = PMXAgent()


# def chat_with_agent(question: str):
#     initial_state = {"question": question}
#     # Ejecutar el grafo
#     final_state = an_agent.graph.invoke(initial_state, config)
#     return final_state["answer"]


# Ejemplo de uso
# response1 = chat_with_agent("Me llamo Ernesto. ¿Cómo evalúo a un empleado?")
# print("Agente:", response1)

# response2 = chat_with_agent("Cual es mi nombre?")
# print("Agente:", response2)

# response3 = chat_with_agent("Donde visualizo las opiniones cliente proveedor?")
# print("Agente:", response3)


import gradio as gr


class PMXAgent_gui:
    def __init__(self, graph, share=False):
        self.graph = graph
        self.share = share
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        # self.sdisps = {} #global
        self.demo = self.create_interface()

    def run_agent(self, question, history):

        response = self.graph.invoke({"question": question}, self.thread)
        return {"role": "assistant", "content": response["answer"]}

    def create_interface(self):
        return gr.ChatInterface(fn=self.run_agent, type="messages")

    def launch(self, share=None):
        # seteo el thread para la memoria del agent
        self.thread_id += 1  # new agent, new thread
        self.threads.append(self.thread_id)
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}

        if port := os.getenv("PORT1"):
            self.demo.launch(share=True, server_port=int(port), server_name="0.0.0.0")
        else:
            self.demo.launch(share=self.share)


agenteAgente = PMXAgent()
app = PMXAgent_gui(agenteAgente.graph)
app.launch()