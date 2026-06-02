from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import json

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

prompt = PromptTemplate(
    input_variables=["event"],
    template="""
You are a strict security AI agent.

Analyze the event and respond ONLY in valid JSON format.

Return:
{
  "alert_level": "low | medium | high",
  "action": "notify | ignore"
}

Event:
{event}
"""
)

def decide(event):
    response = llm.invoke(prompt.format(event=event))
    return json.loads(response.content)
