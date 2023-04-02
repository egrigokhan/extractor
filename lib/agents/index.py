import os

from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.utilities import SerpAPIWrapper


class ConversationEndTool(BaseTool):
    name = "Conversation end"
    description = "call this when the conversation ends. The action_input is the summary of the conversation."

    def _run(self, query: str) -> str:
        """Use the tool."""
        return query

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


chat = ChatOpenAI(temperature=0)


class ChatTool(BaseTool):
    name = "Chat"
    description = ""

    def _run(self, query: str, messages) -> str:
        """Use the tool."""

        print(messages)

        return chat(messages)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")
