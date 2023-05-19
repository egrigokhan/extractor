import os

from langchain import ConversationChain, LLMChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

def approve(type, request):
    #  switch case
    return "Error approving."


def run(message, history):

    chat = ChatOpenAI(temperature=0)

    messages = [
        SystemMessage(
            content="""
      You are InformationExtractorGPT. Your job is to parse information out of the user's text based on the following instructions:

      Instructions: """ + os.environ["instructions"] + """

      Respond only with JSON in the following format:

      {{
        fields: {{
            "field_1": "Field 1",
            "field_2": "Field 2",
            "field_3": "Field 3",
            ...
        }},
        json: {{
            "field_1": ...,
            "field_2": ...,
            "field_3": ...,
            ...
        }},
        error: "Error message" || null
      }}
      """)
    ]

    messages.append(HumanMessage(content=message))

    res = chat(messages)

    return res.content


def setup(config):
    try:
        os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
        os.environ["instructions"] = config["instructions"]
        print("Setup complete.")
    # print error
    except KeyError:
        print("Error setting up.")
        exit()

