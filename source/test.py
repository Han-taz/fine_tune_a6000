from dotenv import load_dotenv

load_dotenv("/home/kevin/projects/langgraph/.env")
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
memory = MemorySaver()
import psycopg

conn_string = "postgresql://postgres:qwe123@localhost:5432/postgres"

drop_tables_sql = """
DROP TABLE IF EXISTS checkpoints CASCADE;
DROP TABLE IF EXISTS checkpoint_blobs CASCADE;
DROP TABLE IF EXISTS checkpoint_writes CASCADE;
"""

create_tables_sql = """
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    checkpoint JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id),
    UNIQUE (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, channel, version),
    FOREIGN KEY (thread_id, checkpoint_ns, checkpoint_id)
        REFERENCES checkpoints (thread_id, checkpoint_ns, checkpoint_id)
        ON DELETE CASCADE
);

CREATE TABLE checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx),
    FOREIGN KEY (thread_id, checkpoint_ns, checkpoint_id)
        REFERENCES checkpoints (thread_id, checkpoint_ns, checkpoint_id)
        ON DELETE CASCADE
);
"""

try:
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(drop_tables_sql)
            cur.execute(create_tables_sql)
        conn.commit()
    print("테이블이 성공적으로 생성되었습니다.")
except Exception as e:
    print(f"테이블 생성 중 오류 발생: {e}")

import psycopg

# 연결 문자열
conn_string = "postgresql://postgres:qwe123@localhost:5432/postgres"

# psycopg Connection 객체 생성
conn = psycopg.connect(conn_string)

# PostgresSaver 인스턴스 생성
pgs_memory = PostgresSaver(conn=conn)

from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.google_serper import GoogleSerperResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = GoogleSerperResults(k=2)
tools = [tool]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=pgs_memory)


config = {"configurable": {"thread_id": "999"}}
user_input = "오늘 서울 날씨는(흐린지, 맑은지, 비오는지) 어떻고 오늘 날짜는 몇일인가요?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()