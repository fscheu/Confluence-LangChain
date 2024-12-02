import functools
import re
import operator
from typing import Annotated, Sequence, TypedDict, List

from colorama import Fore, Style
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper

from agent_tester_prompt import TESTER_PROMPT, JIRA_TESTER_PROMPT

from setup_environment import set_environment_variables

set_environment_variables("Agent_Tester")

from langgraph.checkpoint.sqlite import SqliteSaver
from contextlib import ExitStack

stack = ExitStack()
memory = stack.enter_context(SqliteSaver.from_conn_string(":memory:"))

from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)


class TestCase(BaseModel):
    ticket_rq: str
    summary: str
    description: str
    steps: str
    expected_result: str


class JiraTicket(BaseModel):
    key: str = Field(description="The key of the ticket")
    summary: str = Field(description="The summary of the ticket")
    # description: str
    # confluence_link: str
    # confluence_content: str
    status: str = Field(description="The status of the ticket")


class AgentState(TypedDict):
    task: str
    ticket_list: List[JiraTicket]
    testcase_list: List[TestCase]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


SUPERVISOR_AGENT_NAME = "test_supervisor"
JIRA_AGENT_NAME = "jira_agent"
CONFLUENCE_AGENT_NAME = "confluence_agent"
TEST_WRITER_NAME = "test_case_agent"

# members = [JIRA_AGENT_NAME, CONFLUENCE_AGENT_NAME, TEST_WRITER_NAME]
members = [JIRA_AGENT_NAME]
options = ["__humano__", "__end__"] + members


def create_agent(llm: BaseChatModel, tools: list, system_prompt: str):
    agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)
    return agent_executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


router_function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

team_supervisor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", TESTER_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Dada la conversación anterior, quien debe responder a continuación?"
            " O debería __end__? Selecciona una de las : {options}",
        ),
    ]
).partial(options=", ".join(options), members=", ".join(members))

team_supervisor_chain = (
    team_supervisor_prompt_template
    | model.bind_functions(functions=[router_function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)
tools = toolkit.get_tools()

# Convert JiraAction tools to OpenAI-compatible tools
openai_compatible_tools = []
for tool in tools:
    # Generate a sanitized tool name
    sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", tool.name)
    tool.name = sanitized_name
    openai_compatible_tools.append(tool)

jira_agent = create_agent(model, openai_compatible_tools, JIRA_TESTER_PROMPT)

# jira_agent = create_agent(model, toolkit.get_tools(), JIRA_TESTER_PROMPT)
jira_agent_node = functools.partial(agent_node, agent=jira_agent, name=JIRA_AGENT_NAME)


workflow = StateGraph(AgentState)
workflow.add_node(SUPERVISOR_AGENT_NAME, team_supervisor_chain)
workflow.add_node(JIRA_AGENT_NAME, jira_agent_node)
# workflow.add_node(CONFLUENCE_AGENT_NAME, )

workflow.set_entry_point(SUPERVISOR_AGENT_NAME)
# for member in members:
#     workflow.add_edge(member, SUPERVISOR_AGENT_NAME)
workflow.add_edge(JIRA_AGENT_NAME, SUPERVISOR_AGENT_NAME)

conditional_map = {name: name for name in members}
conditional_map["__end__"] = "__end__"
workflow.add_conditional_edges(
    SUPERVISOR_AGENT_NAME, lambda x: x["next"], conditional_map
)
tester_agent_graph = workflow.compile()

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
