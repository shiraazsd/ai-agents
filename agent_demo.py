import os
import logging
import subprocess
from typing import List, Optional
from pathlib import Path
import requests
import time
import uuid

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import Tool
import textwrap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("agent")
ctx_logger = logging.getLogger("llm_context")

# Config
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
TOOLS_BASE_URL = os.getenv("TOOLS_BASE_URL", "http://127.0.0.1:8000")

# Globals
#_embeddings: Optional[OllamaEmbeddings] = None
#_retriever = None

class ToolLogHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "unknown_tool")
        logger.info(f"[TOOL START] {name} input={input_str!r}")
    def on_tool_end(self, output, **kwargs):
        preview = (output or "")[:200].replace("\n", " ")
        logger.info(f"[TOOL END] output={preview!r}")

class ContextLogHandler(BaseCallbackHandler):
    """
    Logs the exact message context passed to the LLM each time it is invoked.
    Truncates long contents for readability; set FULL_CONTEXT=1 to disable truncation.
    """
    def __init__(self, max_chars: int = 300):
        self.max_chars = max_chars
        self.full = os.getenv("FULL_CONTEXT", "0") == "1"
    def on_chat_model_start(self, serialized, messages, **kwargs):
        # messages: List[List[BaseMessage]] (batch usually size 1)
        batch = messages[0] if messages else []
        total_chars = sum(len(getattr(m, "content", "") or "") for m in batch)
        lines = [f"[CTX START] LLM call at {time.strftime('%H:%M:%S')} (messages={len(batch)}, total_chars={total_chars})"]
        for i, m in enumerate(batch, 1):
            role = m.type
            content = getattr(m, "content", "")
            display = content if self.full else (content[: self.max_chars] + ("..." if len(content) > self.max_chars else ""))
            one_line = textwrap.shorten(display.replace("\n", " ⏎ "), width=self.max_chars + 20, placeholder="...")
            lines.append(f"  {i:02d} {role:<9} | {one_line}")
        ctx_logger.info("\n".join(lines))
    def on_llm_end(self, response, **kwargs):
        try:
            generations = response.generations[0]
            if generations:
                txt = generations[0].text
                display = txt if self.full else (txt[: self.max_chars] + ("..." if len(txt) > self.max_chars else ""))
                ctx_logger.info(f"[CTX END] LLM output (truncated={not self.full}): {display.replace(chr(10),' ⏎ ')}")
        except Exception as e:
            ctx_logger.debug(f"[CTX END] Unable to log output: {e}")

def _http_post(path: str, json: dict) -> str:
    url = f"{TOOLS_BASE_URL}{path}"
    req_id = uuid.uuid4().hex[:8]
    start = time.time()
    logger.info(f"[HTTP POST -> {url}] id={req_id} payload={json}")
    try:
        r = requests.post(url, json=json, timeout=60)
        elapsed = (time.time() - start) * 1000
        try:
            data = r.json()
        except Exception:
            data = {"result": f"(non-JSON body len={len(r.text)})"}
        result_text = data.get("result", "")
        preview = (result_text[:300] + ("..." if len(result_text) > 300 else ""))
        logger.info(f"[HTTP POST <- {url}] id={req_id} status={r.status_code} ms={elapsed:.1f} result_preview={preview!r}")
        return data.get("result", f"Error: bad response {r.status_code}")
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error(f"[HTTP POST !! {url}] id={req_id} ms={elapsed:.1f} error={e}")
        return f"Error: HTTP POST {url} failed: {e}"

def _http_get(path: str, params: dict) -> str:
    url = f"{TOOLS_BASE_URL}{path}"
    req_id = uuid.uuid4().hex[:8]
    start = time.time()
    logger.info(f"[HTTP GET  -> {url}] id={req_id} params={params}")
    try:
        r = requests.get(url, params=params, timeout=30)
        elapsed = (time.time() - start) * 1000
        try:
            data = r.json()
        except Exception:
            data = {"result": f"(non-JSON body len={len(r.text)})"}
        result_text = data.get("result", "")
        preview = (result_text[:300] + ("..." if len(result_text) > 300 else ""))
        logger.info(f"[HTTP GET  <- {url}] id={req_id} status={r.status_code} ms={elapsed:.1f} result_preview={preview!r}")
        return data.get("result", f"Error: bad response {r.status_code}")
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error(f"[HTTP GET  !! {url}] id={req_id} ms={elapsed:.1f} error={e}")
        return f"Error: HTTP GET {url} failed: {e}"

rag_refresh_tool = Tool(
    name="rag_refresh",
    description="Rebuild the PDF index. Optional pdf_dir.",
    func=lambda pdf_dir="": _http_post("/rag/refresh", {"pdf_dir": pdf_dir}),
)

rag_search_tool = Tool(
    name="rag_search",
    description="Search PDFs. Args: query (str), k (int optional).",
    func=lambda query, k=5: _http_post("/rag/search", {"query": query, "k": k}),
)

rag_list_pdfs_tool = Tool(
    name="rag_list_pdfs",
    description="List indexed PDF files (optional folder).",
    func=lambda folder="": _http_get("/rag/list", {"folder": folder}),
)

shell_cmd_tool = Tool(
    name="shell_cmd",
    description="Run safe shell command (ls, pwd, df, echo). Arg: command.",
    func=lambda command: _http_post("/shell", {"command": command}),
)

def main():
    print("LangGraph ReAct Agent (local Ollama + RAG). Type 'exit' to quit.")
    print(f"Using model: {OLLAMA_MODEL}")
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)

    tools = [rag_refresh_tool, rag_search_tool, rag_list_pdfs_tool, shell_cmd_tool]
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "SYSTEM ROLE:\n"
            "You are a precise assistant that can call the following tools:\n"
            "1) rag_search(query: str, k: int=5) -> search indexed PDF chunks.\n"
            "2) rag_refresh(pdf_dir?: str) -> (re)build the PDF index (only if missing or clearly outdated).\n"
            "3) rag_list_pdfs(folder?: str) -> list available PDFs.\n"
            "4) shell_cmd(command: str) -> run a SAFE shell command (only: ls, pwd, df, echo).\n"
            "\n"
            "WHEN TO USE TOOLS:\n"
            "- Any question asking about, summarizing, locating, extracting, quoting, or verifying information from PDFs: FIRST call rag_search.\n"
            "- If rag_search returns that no index exists or similar, THEN call rag_refresh ONCE, followed immediately by rag_search.\n"
            "- If user asks what PDFs exist: call rag_list_pdfs.\n"
            "- Only use shell_cmd when user explicitly requests an allowed shell action relevant to listing or environment context.\n"
            "- Never fabricate PDF contents; always rely on rag_search output.\n"
            "\n"
            "OUTPUT / FORMAT FOR TOOL INVOCATION (STRICT):\n"
            "You must use EXACTLY this ReAct pattern for each step requiring a tool:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: <JSON-serializable arguments, object or primitive>\n"
            "\n"
            "Example (searching PDFs):\n"
            "Thought: I should search the PDFs for onboarding timeline.\n"
            "Action: rag_search\n"
            "Action Input: {\"query\": \"onboarding timeline\", \"k\": 5}\n"
            "\n"
            "After a tool result arrives you will receive an Observation message. Then either:\n"
            "1) Decide another tool is needed (repeat Thought/Action/Action Input), OR\n"
            "2) Conclude with a final answer:\n"
            "Thought: I can now answer.\n"
            "Final Answer: <concise answer with citations like filename.pdf#p3>\n"
            "\n"
            "CITATIONS:\n"
            "- For each factual PDF-derived claim include at least one citation in the form (filename.pdf#pX).\n"
            "- If no information found: clearly state no relevant information was found.\n"
            "\n"
            "GUARDRAILS:\n"
            "- Do NOT call rag_refresh unless index missing / explicitly user requests rebuild.\n"
            "- Do NOT invent tool names or parameters.\n"
            "- Action Input MUST be valid JSON (no trailing commas, use double quotes for keys/strings).\n"
            "- Only one Action per step.\n"
            "- Do NOT wrap Action / Action Input in code fences.\n"
            "- If user asks multiple PDF questions in one turn, a single rag_search with a broad query is acceptable unless follow-ups needed.\n"
            "\n"
            "If the user asks something unrelated to PDFs and no tool is required, respond directly with:\n"
            "Thought: I can answer without tools.\n"
            "Final Answer: <answer>\n"
        ),
    )

    # Keep full chat history (includes tool traces)
    messages: List[BaseMessage] = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        messages.append(HumanMessage(content=user_input))
        result = agent.invoke(
            {"messages": messages},
            config={"callbacks": [ToolLogHandler(), ContextLogHandler()], "recursion_limit": 8},
        )
        messages = result["messages"]
        ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if ai:
            print("Agent:", ai.content)

if __name__ == "__main__":
    main()
