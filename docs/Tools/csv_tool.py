from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI

def csv(csv_path, query):
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09"),#use gpt 3.5 turbo for the faster responese from this tool.
        csv_path,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    response = agent.run(query)
    return response
