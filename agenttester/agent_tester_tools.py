import operator
from typing import Annotated, TypedDict, List
from colorama import Fore, Style


from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from contextlib import ExitStack

from pydantic import BaseModel, Field

from jira import JiraTester, JiraTicket

from agent_tester_prompt import TESTER_PROMPT_TOOLS, WRITER_PROMPT_TOOLS

from setup_environment import set_environment_variables

set_environment_variables("Agent_Tester_Tools")


stack = ExitStack()
memory = stack.enter_context(SqliteSaver.from_conn_string(":memory:"))

from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)


class TestCase(BaseModel):
    ticket_rq: str = Field(description="El key del ticket analizado para generar los casos de test")
    titulo: str = Field(description="El titulo del caso de test")
    descripcion: str = Field(description="La descripcion del caso de test")
    pasos: str = Field(description="Los pasos a realizar para ejecutar el caso de test")
    resultado_esperado: str = Field(description="El resultado esperado al ejecutar el caso de test")


# define a class as a list of TestCase
from typing import List

class TestCaseList(BaseModel):
    test_cases: List[TestCase]


class AgentState(TypedDict):
    ticket_list: List[JiraTicket]
    testcase_list: List[TestCase]
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


jira = JiraTester()


@tool
def get_jira_tickets(project: str):
    # use jira to execute a jql wich gets the tickets from param project and status 'ESTIMACION Y PLANIFICACION'
    """
    Use JIRA to execute a JQL query which gets the tickets from the provided project
    and status 'ESTIMACION Y PLANIFICACION'

    Args:
        project (str): the project to search for

    Returns:
        A list of JiraTickets that match the query
    """

    jql_str = f'project = {project} AND status = "ESTIMACIóN Y PLANIFICACIóN"'
    tickets = jira.search(jql_str)
    return {"ticket_list": tickets}  # return the tickets


@tool
def get_confluence_definitions(ticket_list: List[JiraTicket]):
    """
    Use Confluence to get the definitions for the tickets provided.
    Stores the definitions externally and returns a reference.
    """
    # Get the definitions for the tickets
    ticket_list = jira.get_confluence_definitions(ticket_list)

    return {"ticket_list": ticket_list}


@tool
def test_case_agent(ticket_list: List[JiraTicket]):
    """
    Use the LLM model to generate test cases for each ticket in the list.
    """
    structured_llm = model.with_structured_output(TestCaseList)
    testcase_list = []
    for ticket in ticket_list:
        # Get the confluence content for the ticket
        confluence_content = ticket.confluence_content
        # Load the file containing the confluence content
        with open(confluence_content, 'r') as f:
            content = f.read()
        # Use the LLM model to generate test cases
        prompt = WRITER_PROMPT_TOOLS.format(content=content)
        test_cases = structured_llm.invoke(prompt)
        # Create a TestCase object for each test case
        testcase_list += test_cases.test_cases
    return {"testcase_list": testcase_list}


tools = [get_jira_tickets, get_confluence_definitions, test_case_agent]

tester_agent_graph = create_react_agent(model, tools, state_modifier=TESTER_PROMPT_TOOLS)

for chunk in tester_agent_graph.stream(
    {
        "messages": [
            HumanMessage(content="Genera los casos de test para el proyecto CUX")
        ]
    }
):
    if "__end__" not in chunk:
        print(chunk)
        print(f"{Fore.GREEN}#############################{Style.RESET_ALL}")
