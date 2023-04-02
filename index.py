import os

from langchain import ConversationChain, LLMChain, OpenAI, PromptTemplate
from langchain.agents import Tool, initialize_agent, AgentExecutor
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory)
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from lib.agents.index import ChatTool, ConversationEndTool
from lib.email.index import send_email_with_summary

chat = ChatOpenAI(temperature=0)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)


def run(message, history):
    tools = [
        Tool(
            name="Never",
            description="never call this tool. if you do call it as action: 'Never', action_input: 'Never'",
            func=lambda x: send_email_with_summary(history),
            return_direct=True
        ),
    ]

    llm = ChatOpenAI(temperature=0)

    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=f"""You are UserInterviewGPT.
        
        UserInterviewGPT is designed to hold a chat with a user and conduct user interviews about a specific product, learn their pain points, learn what the product can offer them, and report back to the founders of the project.

        UserInterviewGPT is required to keep the conversation light and to the point, and its priority should be to delve into the value the product can bring to the user. Initially UserInterviewGPT must make sure to learn about the user's pain points, and only towards the end of the conversation talk to them about the product.
        
        After the conversation ends, UserInterviewGPT must report its takeaways from this conversation to the founders using the Email tool, but only after the conversation ends.
        
        Here are the product details: 
        
        Name: {os.environ['name']} 
        Tagline: {os.environ['tagline'] } 
        Description: {os.environ['description'] }""",
        human_message="""TOOLS
            ------
            UserInterviewGPT can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:
            {{tools}}
            {format_instructions}
            USER'S INPUT
            --------------------
            Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):
            {{{{input}}}}"""
    )
    # res = agent(input=message)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True)

    res = agent_executor.run({
        "input": message,
        "chat_history": history[-6:],
    })

    return res


def setup(config):
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["name"] = config["name"]
    os.environ["tagline"] = config["tagline"]
    os.environ["description"] = config["description"]
