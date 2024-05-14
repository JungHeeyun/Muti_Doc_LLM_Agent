from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI

def csv(csv_path, query):
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4o-2024-05-13"),# updated to gpt-4o-2024-05-13 model 
        csv_path,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    response = agent.run(query)
    return response
