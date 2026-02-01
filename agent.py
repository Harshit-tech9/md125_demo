from typing import TypedDict, List 
from langgraph.graph import StateGraph, START, END  
from langchain_groq import ChatGroq 
from langchain.schema import HumanMessage, SystemMessage, AIMessage    
import os

# --- LLM ---
llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", api_key=grok_api_key) 

# print(llm.invoke("Hello, how are you?").content)  
# ---STATE ---  

class AgentState(TypedDict):
    messages: List  

# --- SIMPLE TOOL --- 

def calculator_tool(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"
    
# --- NODE 1: LLM Chat mode --- 

def chat_node(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    return {"messages": messages}

# --- Router: decides next node (must return node name string, not state) ---
def tool_router(state: AgentState) -> str:
    last_message = state["messages"][-1]
    content = getattr(last_message, "content", "") or ""
    if isinstance(content, str) and content.strip().lower().startswith("calculate"):
        return "calculator"
    return "chat"  # continue to END via edge "chat" -> END 


# --- NODE 3: Calculator Tool Node  --- 
def calculator_tool_node(state: AgentState) -> AgentState: 
    last_user_message = state["messages"][-1].content  
    expression = last_user_message.replace("calculate ", " ").strip() 
    result = calculator_tool(expression) 
    state["messages"].append(AIMessage(content=f"The result is {result}"))  
    return {"messages": state["messages"]}  

# --- Build the Graph ---  
builder = StateGraph(AgentState)  
builder.add_node("chat", chat_node) 
builder.add_node("calculator", calculator_tool_node)  

builder.add_conditional_edges(
    "chat",
    tool_router,
    {
        "calculator": "calculator",
        "chat": END
    }
)

builder.set_entry_point("chat") 

graph = builder.compile()

# Save graph diagram as PNG file
# png_bytes = graph.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png_bytes)
# print("Graph saved to graph.png")

# print(graph.invoke({"messages": [HumanMessage(content="Hello, how are you?")]}))  

# # --- Run the Graph ---  

# print("--------------------------------")
# print("Running the Graph...")


# result = graph.invoke({"messages": [HumanMessage(content="Calculate 1+1")]})
# print(result)  

def run_agent(user_input: str, history: list) -> list:
    """Run the agent with user input and message history. Returns updated list of messages (for session_state.history)."""
    messages = list(history) if history else []
    messages.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": messages})
    return result["messages"]


def is_calculator_query(content: str) -> bool:
    """True if the user message should go to the calculator tool."""
    return isinstance(content, str) and content.strip().lower().startswith("calculate")


def stream_chat_tokens(messages: list):
    """Yield LLM response tokens one by one. Use for streaming in the UI."""
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content