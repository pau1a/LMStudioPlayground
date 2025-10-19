from __future__ import annotations
import os, sys, json, math, re, time, traceback, argparse
from typing import Any, Dict, Callable, Optional, List
from openai import OpenAI

# ===== CONFIG =====
LM = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
MODEL = "mixtral-latest"
ROOT  = os.path.abspath(os.getcwd())

ENABLE_DETERMINISTIC = True
ENABLE_BOOTSTRAP = True
FORCE_AGENT_PREFIX = "agent:"   # bypass layers 1 & 2 when prompt starts with this

# ===== COLOUR LOGGING =====
C = type("C", (), {
    "RST":"\x1b[0m","DIM":"\x1b[2m","B":"\x1b[1m","GRN":"\x1b[32m","YEL":"\x1b[33m",
    "BLU":"\x1b[34m","MAG":"\x1b[35m","CYN":"\x1b[36m","RED":"\x1b[31m"
})()
def ts(): return time.strftime("%H:%M:%S")
def log(k,m):
    col={"SYS":C.DIM,"USR":C.BLU,"LLM":C.MAG,"ACT":C.CYN,"TOOL":C.YEL,
         "RES":C.GRN,"PARSE":C.DIM,"WARN":C.RED}.get(k,C.DIM)
    print(f"{col}[{ts()} {k}] {m}{C.RST}",flush=True)

# ===== REGEX =====
SAFE_EXPR = re.compile(r"^[0-9+\-*/().\s^]*$")
NUM_RE  = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
FILE_RE = re.compile(r"(?:\./|/)?[^\s\"']*?\.txt", re.IGNORECASE)
READ_INTENT = re.compile(r"\b(what\s+is\s+in|show|display|print|read|open)\b", re.IGNORECASE)

# ===== TOOLS =====
def read_file(path:str)->str:
    p=os.path.abspath(os.path.join(ROOT,path))
    if not p.startswith(ROOT): raise ValueError("path outside project")
    if not os.path.isfile(p): raise ValueError(f"not a file: {p}")
    with open(p,"r",encoding="utf-8",errors="ignore") as f:
        return f.read()[:8000]

def write_file(path:str,text:str)->str:
    p=os.path.abspath(os.path.join(ROOT,path))
    if not p.startswith(ROOT): raise ValueError("path outside project")
    with open(p,"w",encoding="utf-8") as f: f.write(text)
    return f"[wrote {len(text)} chars to {path}]"

def calc(expr:str)->str:
    if not SAFE_EXPR.match(expr): raise ValueError("bad chars in expr")
    expr=expr.replace("^","**")
    return str(eval(expr,{"__builtins__":{}},{"math":math}))

def find_number(text:str)->str:
    m=NUM_RE.search(text)
    if not m: raise ValueError("no number found")
    return m.group(0)

TOOLS={"read_file":read_file,"write_file":write_file,"calc":calc,"find_number":find_number}

# ===== SYSTEM PROMPT =====
SYSTEM=r"""
You are a programmatic agent.
You may call only these tools:
- read_file(path)
- write_file(path, text)
- calc(expr)
- find_number(text)

Rules:
- Reply with ONE JSON object ONLY:
  {"tool":"<name>","args":{...}}  OR  {"final":"<message>"}
- Use a tool only when needed. For general questions, answer directly with {"final":"..."}.
- After a tool call, you will receive: TOOL_RESULT: <data>
  Then call another tool or finish with {"final":"..."}.
- No prose, no markdown, no multiple objects.
- Do NOT invent tools.
"""

# ===== CORE HELPERS =====
def llm(msgs):
    out=LM.chat.completions.create(model=MODEL,messages=msgs,temperature=0.0,
                                   max_tokens=700,response_format={"type":"text"}).choices[0].message.content or ""
    log("LLM",out[:400].replace("\n"," ")[:400]+("..." if len(out)>400 else ""))
    return out

def extract_last_json_dict(text):
    start=-1;depth=0;last=None
    for i,ch in enumerate(text):
        if ch=="{":
            if depth==0:start=i
            depth+=1
        elif ch=="}":
            if depth>0:
                depth-=1
                if depth==0 and start!=-1:
                    try:
                        j=json.loads(text[start:i+1])
                        if isinstance(j,dict): last=j
                    except: pass
    return last

def run_tool(n,a):
    log("ACT",f"{n} {a}")
    try:
        r=str(TOOLS[n](**a))
        log("TOOL",f"{len(r)} chars")
        return r
    except Exception as e:
        return f"ERROR: {e}"

def _first_path_in(t):
    if not t:return None
    m=FILE_RE.search(t)
    return m.group(0).rstrip('.,!?:;\'")') if m else None

def _is_read_intent(text: str) -> bool:
    return bool(READ_INTENT.search(text or ""))

# ===== LAYERS =====
def deterministic_execute(p):
    if not ENABLE_DETERMINISTIC: return None
    if re.search(r"\bwhat\s+is\s+in\b",p,re.I) and ".txt" in p:
        path=_first_path_in(p)
        if path:
            try:return read_file(path)
            except Exception as e:return f"[error: {e}]"
    return None

def _bootstrap_file_read(msgs,path):
    if not ENABLE_BOOTSTRAP: return ""
    log("SYS",f"bootstrap read_file {path}")
    r=run_tool("read_file",{"path":path})
    msgs.append({"role":"assistant","content":json.dumps({"tool":"read_file","args":{"path":path}})})
    msgs.append({"role":"system","content":f"TOOL_RESULT: {r}"})
    return r

# ===== AGENT CORE =====
def run_query(q):
    forced_agent = q.lower().startswith(FORCE_AGENT_PREFIX)
    if forced_agent:
        q = q[len(FORCE_AGENT_PREFIX):].lstrip()
        log("SYS","FORCED agent mode (skipping deterministic & bootstrap)")

    det=None
    if ENABLE_DETERMINISTIC and not forced_agent:
        det=deterministic_execute(q)
        if det is not None:
            log("RES",f"deterministic -> {det[:80]}...")
            return det

    msgs=[{"role":"system","content":SYSTEM.strip()},
          {"role":"user","content":q}]

    if ENABLE_BOOTSTRAP and not forced_agent:
        path=_first_path_in(q)
        if (path is not None) or (" file" in q.lower()):
            if path is None and os.path.isfile(os.path.join(ROOT,"notes.txt")):
                path="./notes.txt"
            if path:
                _bootstrap_file_read(msgs,path)

    last_tool_result=None
    for _ in range(40):
        raw=llm(msgs).strip()
        data=extract_last_json_dict(raw)
        if not data:
            msgs.append({"role":"user","content":
                "Return ONE JSON object only: {'tool':..., 'args':...} OR {'final':'...' }."})
            continue

        tool=data.get("tool")
        args=data.get("args") if isinstance(data.get("args"),dict) else None
        final=data.get("final")

        if tool and args is not None:
            msgs.append({"role":"assistant","content":json.dumps({"tool":tool,"args":args})})
            result=run_tool(tool,args)
            last_tool_result=result
            msgs.append({"role":"system","content":f"TOOL_RESULT: {result}"})
            continue

        if final is not None:
            if _is_read_intent(q) and last_tool_result is not None:
                log("RES",f"final (tool-trusted) -> {last_tool_result[:120]}...")
                return last_tool_result if last_tool_result else "[empty file]"
            log("RES",f"final -> {final}")
            return str(final)

    log("WARN","loop limit reached")
    return "ERROR: exceeded tool loop limit"

def plain_chat(p):
    msgs=[{"role":"system","content":"You are a helpful assistant."},
          {"role":"user","content":p}]
    r=llm(msgs).strip()
    log("RES",f"chat -> {r[:120]}...")
    return r

# ===== CLI =====
if __name__=="__main__":
    print("Agent with force-agent prefix. Ctrl-C to quit.\n")
    while True:
        try:q=input("You > ").strip()
        except (EOFError,KeyboardInterrupt):break
        if not q:continue
        if q.lower() in {"quit","exit"}:break
        r=run_query(q)
        if r.strip().lower()=="i cannot do that":
            r=plain_chat(q)
        print("AI >",r)
