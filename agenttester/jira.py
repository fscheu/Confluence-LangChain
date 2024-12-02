import json
import re
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from bs4 import BeautifulSoup as bs
from langchain_community.utilities.jira import JiraAPIWrapper


class JiraTicket(BaseModel):
    key: str = Field(description="The key of the ticket")
    summary: str = Field(description="The summary of the ticket")
    # description: str
    confluence_link: str = Field(description="Link to Confluence page with definition")
    confluence_content: Optional[str] = None
    status: str = Field(description="The status of the ticket")


class JiraTester(JiraAPIWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_issues(self, issues: Dict) -> List[JiraTicket]:
        parsed = []
        for issue in issues["issues"]:
            key = issue["key"]
            summary = issue["fields"]["summary"]
            status = issue["fields"]["status"]["name"]
            rel_links = self.jira.get_issue_remote_links(key)
            for link in rel_links:
                if link["application"]["name"] == "Confluence":
                    confluence_link = link["object"]["url"]
            ticket = JiraTicket(
                key=key, summary=summary, status=status, confluence_link=confluence_link
            )
            parsed.append(ticket)
        return parsed

    def parse_html_content(self, html_content):
        # Parse HTML content
        soup = bs(html_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text

    def search(self, query: str) -> list:
        issues = self.jira.jql(query)
        parsed_issues = self.parse_issues(issues)
        return parsed_issues

    def save_data(
        self, ticket: JiraTicket, prefix="confluence_definition", page_id: str = None
    ) -> None:

        conf_content = (
            self.confluence.get_page_by_id(page_id.group(1), expand="body.export_view")
            .get("body")
            .get("export_view")
            .get("value")
        )
        conf_content = self.parse_html_content(conf_content)
        file_name = f"{prefix}_{ticket.key}.json"

        # Save the definition to the file
        with open(file_name, "w") as file:
            json.dump(conf_content, file)  # Save the entire ticket as JSON

        # Update the JiraTicket object with the file name
        ticket.confluence_content = file_name

    def get_confluence_definitions(
        self, ticket_list: List[JiraTicket]
    ) -> List[JiraTicket]:
        """
        For each ticket in the list, fetch the corresponding Confluence page id from the
        ticket's remote link. If the page id is found, fetch the content of the page using
        the Confluence API and store it in the ticket's confluence_content attribute.
        If the page id is not found, return an Exception

        Args:
            ticket_list (List[JiraTicket]): The list of JiraTickets to fetch the Confluence
                page content for

        Returns:
            List[JiraTicket]: The same list of JiraTickets with the confluence_content
                attribute populated
        """
        for ticket in ticket_list:
            page_id = re.search(
                r"https://portal-sede-techint\.atlassian\.net/wiki/pages/viewpage\.action\?pageId=([^&]+)",
                ticket.confluence_link,
            )
            if page_id:
                self.save_data(ticket, prefix="confluence_definition", page_id=page_id)
            else:
                return Exception("No page id found")
        return ticket_list
