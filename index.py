import os

from langchain import ConversationChain, LLMChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0)


def approve(type, request):
    #  switch case
    return "Error approving."


def run(message, history):

    messages = [
        SystemMessage(
            content="""
      You are InformationExtractorGPT. Your job is to parse information out of the user's text based on the following instructions:

      Instructions: """ + os.environ["instructions"] + """

      Respond only with markdown-formatted text. 
      """)
    ]

    messages.append(HumanMessage(content=message))

    res = chat(messages)

    return res.content


def setup(config):
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["instructions"] = config["instructions"]
