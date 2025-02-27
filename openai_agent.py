from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph  # ✅ Use StateGraph for managing state
from langgraph.graph.message import add_messages  
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# ✅ Define the correct state structure
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize the LLM
model = ChatOpenAI(model="gpt-4o-mini")

def make_graph():
    """Simple LLM Agent""" 
    graph_workflow = StateGraph(MessageState)  # ✅ Use MessageState instead of State

    def call_model(state: MessageState):
        """Calls the LLM and returns updated state"""
        return {"messages": [model.invoke(state["messages"])]}
    
    graph_workflow.add_node("model", call_model)
    graph_workflow.add_edge(START, "model")
    graph_workflow.add_edge("model", END)

    agent = graph_workflow.compile()
    return agent

def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: int, b: int) -> int:
        """A simple addition tool"""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])

    def call_model(state: MessageState):
        """Call the model with tools"""
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: MessageState):
        """Check if the last message is a tool call"""
        if state["messages"] and state["messages"][-1].tool_calls:
            return "tools"
        return END
    
    graph_workflow = StateGraph(MessageState)  # ✅ Use MessageState instead of State

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)

    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent

# Create the agent
agent = make_alternative_graph()