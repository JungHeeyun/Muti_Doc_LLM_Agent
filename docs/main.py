from Tools.main_tools import init_tools
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import ToolExecutor
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

def init_app(structure, index, chat_history, faiss_index):
    tools = init_tools(index, faiss_index)
    tool_executor = ToolExecutor(tools)

    # Create the tools to bind to the model
    tools = [convert_to_openai_function(t) for t in tools]

    prompt_template = """
    You are a CMM research assistant dedicated to document analysis developed by the CMM's NLP team.
    Users will mainly ask you questions related to the document. Then you should do your best to help your users using the tools given to you.
    You can call multiple tools simultaneously in parallel.
    You can ask to the user back to clarify the user'query. 

    The types of files submitted to you by users are limited to pdf and csv, but can also be multiple.
    The user's submitted data structure is like this: {structure}

    Potential User Query:
    1. Question about dataset structure: For questions about dataset structure, explain by referring to the ascii dataset tree structure given above. But you can't tell the index number to the user.
    2. Specific concept questions about specific files(Keyword Search): For questions that require searching for specific concepts or keyword about specific files, use the SearchPDF tool.
    3. Question about a specific page of a specific file: When asking a question about a specific page of a specific file, first check above strucuture and pick the right file, if no sure, ask to clarify, when clarified, then use the PDFPage_Analyzer tool.
    4. csv related question: user AskCsv tool.

    Chat History: 
    Only maximum 5 past messages are displayed below.
    {chat_history}
    """
    formatted_prompt = prompt_template.format(structure=structure, chat_history=chat_history)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                formatted_prompt,
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
    model = {"messages": RunnablePassthrough()} | prompt | llm.bind_tools(tools)


    class AgentState(TypedDict):
        messages: Sequence[BaseMessage]


    # Define the function that determines whether to continue or not
    def should_continue(state):
        last_message = state["messages"][-1]
        # If there are no tool calls, then we finish
        if "tool_calls" not in last_message.additional_kwargs:
            return "end"
        # If there is a Response tool call, then we finish
        elif any(
            tool_call["function"]["name"] == "Response"
            for tool_call in last_message.additional_kwargs["tool_calls"]
        ):
            return "end"
        # Otherwise, we continue
        else:
            return "continue"


    # Define the function that calls the model
    def call_model(state):
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": messages + [response]}


    # Define the function to execute tools
    def call_tool(state):
        messages = state["messages"]
        # We know the last message involves at least one tool call
        last_message = messages[-1]

        # We loop through all tool calls and append the message to our message log
        for tool_call in last_message.additional_kwargs["tool_calls"]:
            action = ToolInvocation(
                tool=tool_call["function"]["name"],
                tool_input=json.loads(tool_call["function"]["arguments"]),
                id=tool_call["id"],
            )

            # We call the tool_executor and get back a response
            response = tool_executor.invoke(action)
            # We use the response to create a FunctionMessage
            function_message = ToolMessage(
                content=str(response), name=action.tool, tool_call_id=tool_call["id"]
            )

            # Add the function message to the list
            messages.append(function_message)

        # We return a list, because this will get added to the existing list

        return {"messages": messages}


    # Initialize a new graph
    graph = StateGraph(AgentState)

    # Define the two Nodes we will cycle between
    graph.add_node("agent", call_model)
    graph.add_node("action", call_tool)

    # Set the Starting Edge
    graph.set_entry_point("agent")

    # Set our Contitional Edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    # Set the Normal Edges
    graph.add_edge("action", "agent")

    # Compile the workflow
    app = graph.compile()
    return app 
