###### Funciones de utilidad ######
import os
import configparser
from datetime import date


def read_ini(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


config = read_ini("config.ini")


def set_environment_variables(project_name: str = "") -> None:
    os.environ["OPENAI_API_KEY"] = str(config["LLM"]["OPENAI_API_KEY"])

    os.environ["JIRA_API_TOKEN"] = str(config["JIRA"]["JIRA_API_TOKEN"])
    os.environ["JIRA_USERNAME"] = str(config["JIRA"]["JIRA_USERNAME"])
    os.environ["JIRA_INSTANCE_URL"] = str(config["JIRA"]["JIRA_INSTANCE_URL"])
    os.environ["JIRA_CLOUD"] = str(config["JIRA"]["JIRA_CLOUD"])

    if not project_name:
        project_name = f"Test_{date.today()}"

    os.environ["LANGCHAIN_TRACING_V2"] = str(config["LANG"]["LANGCHAIN_TRACING_V2"])
    os.environ["LANGCHAIN_API_KEY"] = str(config["LANG"]["LANGCHAIN_API_KEY"])
    os.environ["LANGCHAIN_PROJECT"] = project_name
    print("API Keys loaded and tracing set with project name: ", project_name)
