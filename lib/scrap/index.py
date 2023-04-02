
    messages = [
        SystemMessage(
            content="""
      You are UserInterviewGPT. Your job is to conduct user interviews about a specific product, learn their pain points, learn what the product can offer them, and report back to the founders of the project.

      You are required to keep the conversation light and to the point, and your priority should be to delve into the value the product can bring to the user. Initially make sure to learn about the user's pain points, and only towards the end of the conversation talk to them about the product.

      After the conversation ends, report your takeaways from this conversation to the founders through email starting with SUMMARY:

      IMPORTANT: Only send an email after the conversation ends.

      Here are the product details:
      """ + """

      Name:""" + os.environ["name"] + """

      Tagline: """ + os.environ["tagline"] + """

      Description: """ + os.environ["description"])

    ]

    """ messages.extend(history)
    messages += [HumanMessage(content=message)]

    llm = ChatOpenAI(temperature=0)

    tools = [ConversationEndTool()] """
    """ tools.extend([Tool(
        name="Final Answer",
        description="This is the final answer action.",
        func=lambda x: ChatTool()._run(query="", messages=messages)
    )]) """

    """ memory.chat_memory.messages = messages

    agent_chain = initialize_agent(
        tools, llm, agent="chat-conversational-react-description", verbose=False, memory=memory) """


prompt = ConversationalChatAgent.create_prompt(
        tools,
        system_message="""
        You are UserInterviewGPT. Your job is to conduct user interviews about a specific product, learn their pain points, learn what the product can offer them, and report back to the founders of the project.

        You are required to keep the conversation light and to the point, and your priority should be to delve into the value the product can bring to the user. Initially make sure to learn about the user's pain points, and only towards the end of the conversation talk to them about the product.

        After the conversation ends, report your takeaways from this conversation to the founders through email starting with SUMMARY:

        IMPORTANT: Only send an email after the conversation ends.

        Here are the product details:
        """ + """

        Name:""" + os.environ["name"] + """

        Tagline: """ + os.environ["tagline"] + """

        Description: """ + os.environ["description"],
        human_message="",
        input_variables=["agent_scratchpad"]
    )

    llm = ChatOpenAI(temperature=0)
    agent_chain = initialize_agent(
        tools, llm, agent="chat-conversational-react-description", verbose=False, memory=memory)
    # res= agent_chain.run(input=message)