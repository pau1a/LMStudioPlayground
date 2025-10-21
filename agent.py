from __future__ import annotations
import json, os, re, math
from typing import Any, Dict, Optional
from openai import OpenAI

# -----------------------------------------------------------
# BASIC LOCAL CONFIG
# -----------------------------------------------------------
LM = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
MODEL = "qwen2.5-vl-3b-instruct"   # set to the model name you loaded in LM Studio
ROOT  = os.path.abspath(os.getcwd())

PLAN_CONF_THRESHOLD = 0.6
WHITELIST_TOOLS = {"read_file", "write_file", "calc", "find_number"}

# Accept paths like: notes.txt, seft/deg.log, ./foo, foo.bar.gz
PATH_RE = re.compile(r"(?:\./|/)?[A-Za-z0-9._/\-]+\.[A-Za-z0-9]{1,8}")

# Sentinels and direct-command regexes
SENTINEL_AGENT = "<|agent|>"
SENTINEL_CHAT  = "<|chat|>"
RE_READ  = re.compile(r"^!read\s+(.+)$", re.IGNORECASE)
RE_WRITE = re.compile(r"^!write\s+(\S+)\s+<<<\s*(.*)$", re.IGNORECASE | re.DOTALL)
RE_CALC  = re.compile(r"^!calc\s+(.+)$", re.IGNORECASE)
RE_NUM   = re.compile(r"^!num\s+(.+)$",  re.IGNORECASE)

# Persistent session context
known_files: set[str] = set()

# -----------------------------------------------------------
# TOOL IMPLEMENTATIONS
# -----------------------------------------------------------
def read_file(args: Dict[str, Any]) -> str:
    path = args.get("path", "")
    if not path:
        return "ERROR: missing path"
    abs_path = os.path.abspath(os.path.join(ROOT, path))
    if not abs_path.startswith(ROOT):
        return "ERROR: path outside project"
    if not os.path.isfile(abs_path):
        return f"ERROR: file not found: {path}"
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()[:4000]

def write_file(args: Dict[str, Any]) -> str:
    path = args.get("path", "")
    text = args.get("text", "")
    if not path:
        return "ERROR: missing path"
    abs_path = os.path.abspath(os.path.join(ROOT, path))
    if not abs_path.startswith(ROOT):
        return "ERROR: path outside project"
    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(text)
        return f"[wrote {len(text)} chars to {path}]"
    except Exception as e:
        return f"ERROR: {e}"

SAFE_EXPR = re.compile(r"^[0-9+\-*/().\s^]*$")
def calc(args: Dict[str, Any]) -> str:
    expr = str(args.get("expr", ""))
    if not SAFE_EXPR.match(expr):
        return "ERROR: invalid characters"
    expr = expr.replace("^", "**")
    try:
        return str(eval(expr, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"ERROR: {e}"

NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
def find_number(args: Dict[str, Any]) -> str:
    text = str(args.get("text", ""))
    m = NUM_RE.search(text)
    return m.group(0) if m else "ERROR: no number found"

TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "calc": calc,
    "find_number": find_number,
}

# -----------------------------------------------------------
# LLM CORE
# -----------------------------------------------------------
def llm(messages, temperature=0.3, max_tokens=500) -> str:
    resp = LM.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "text"},
    )
    return resp.choices[0].message.content or ""

# -----------------------------------------------------------
# PLANNER
# -----------------------------------------------------------
PLANNER_SYSTEM = (
    "You are a router. Decide how to handle the user's request.\n"
    "Return ONE JSON object only with keys: route, tool, args, confidence.\n"
    "- route: 'tool' or 'chat'\n"
    "- tool: one of ['read_file','write_file','calc','find_number'] or null when route='chat'\n"
    "- args: object of arguments for the chosen tool ({} if none)\n"
    "- confidence: float 0.0..1.0 (your certainty)\n"
    "No prose. No markdown. JSON only."
)

def extract_last_json_dict(text: str) -> Optional[Dict[str, Any]]:
    start = -1
    depth = 0
    last = None
    for i, ch in enumerate(text or ""):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    frag = text[start:i+1]
                    try:
                        obj = json.loads(frag)
                        if isinstance(obj, dict):
                            last = obj
                    except Exception:
                        pass
    return last

def plan_route(user_prompt: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    raw = llm(messages, temperature=0.0, max_tokens=200).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    obj = extract_last_json_dict(raw)
    return obj or {}

def valid_tool_choice(tool: str, args: Dict[str, Any]) -> bool:
    if tool not in WHITELIST_TOOLS:
        return False
    if tool == "read_file" and "path" not in args:
        return False
    if tool == "write_file" and not all(k in args for k in ("path", "text")):
        return False
    if tool == "calc" and "expr" not in args:
        return False
    if tool == "find_number" and "text" not in args:
        return False
    return True

# -----------------------------------------------------------
# HEURISTIC FALLBACK + CONTEXT MEMORY
# -----------------------------------------------------------
def autowrap_to_action(raw: str, user_prompt: str) -> dict:
    raw = (raw or "").strip()
    up  = user_prompt.strip()

    # Detect "X is a file" declarations → remember it
    m = re.findall(r"\b([A-Za-z0-9_\-./]+)\b\s+is\s+a\s+file", up, re.IGNORECASE)
    for name in m:
        known_files.add(name.strip().lower())

    # Direct JSON pass-through
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and ("tool" in data or "final" in data):
            return data
    except Exception:
        pass

    # Detect explicit path-like names with extension
    mp = PATH_RE.search(up)
    if mp:
        path = mp.group(0).rstrip('.,!?:;\'")')
        return {"tool": "read_file", "args": {"path": path}}

    # If user mentions a known file by name (bare filename)
    for name in known_files:
        if re.search(rf"\b{re.escape(name)}\b", up, re.IGNORECASE):
            return {"tool": "read_file", "args": {"path": name}}

    # Bare filename fallback by filesystem probe
    tokens = re.findall(r"[A-Za-z0-9._/\-]+", up)
    for token in tokens:
        if os.path.isfile(os.path.join(ROOT, token)):
            return {"tool": "read_file", "args": {"path": token}}

    # Arithmetic intent: trailing expression
    cm = re.search(r"([-+/*()\s.\d^]+)$", up)
    if cm and re.search(r"\d", cm.group(1)):
        return {"tool": "calc", "args": {"expr": cm.group(1).strip()}}

    return {"final": raw[:4000] or "(no output)"}

# -----------------------------------------------------------
# DIRECT COMMANDS
# -----------------------------------------------------------
def handle_direct_command(q: str) -> Optional[str]:
    m = RE_READ.match(q)
    if m:
        return read_file({"path": m.group(1).strip()})
    m = RE_WRITE.match(q)
    if m:
        path, text = m.group(1), m.group(2)
        return write_file({"path": path, "text": text})
    m = RE_CALC.match(q)
    if m:
        return calc({"expr": m.group(1).strip()})
    m = RE_NUM.match(q)
    if m:
        return find_number({"text": m.group(1)})
    return None

# -----------------------------------------------------------
# RUN LOOP
# -----------------------------------------------------------
def run_query(user_input: str) -> str:
    q = user_input.strip()

    # --- DIRECT COMMANDS (no model) ---
    direct = handle_direct_command(q)
    if direct is not None:
        return direct

    # --- EXPLICIT SENTINELS ---
    force_agent = False
    if q.startswith(SENTINEL_AGENT):
        force_agent = True
        q = q[len(SENTINEL_AGENT):].lstrip()
    elif q.startswith(SENTINEL_CHAT):
        # Force free chat; bypass planner & heuristics
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ]
        return llm(messages, temperature=0.3, max_tokens=600).strip()

    # --- PLANNER PASS ---
    if force_agent:
        plan = plan_route(q)
        route = "tool"  # force tool mode branch consideration
        tool  = plan.get("tool", None)
        args  = plan.get("args", {}) if isinstance(plan.get("args"), dict) else {}
        conf  = 1.0
    else:
        plan = plan_route(q)
        route = str(plan.get("route", "")).lower()
        tool  = plan.get("tool", None)
        args  = plan.get("args", {}) if isinstance(plan.get("args"), dict) else {}
        try:
            conf = float(plan.get("confidence", 0.0))
        except Exception:
            conf = 0.0

    # Confident tool branch
    if route == "tool" and (force_agent or conf >= PLAN_CONF_THRESHOLD) and (tool is None or valid_tool_choice(tool, args)):
        # If planner proposed a specific tool with valid args, run it;
        # otherwise fall back to heuristic inference to build the first action.
        if tool and valid_tool_choice(tool, args):
            fn = TOOLS.get(tool)
            if not fn:
                return f"ERROR: unknown tool '{tool}'"
            result = fn(args)
            return result if tool == "read_file" else f"[TOOL RESULT] {result}"
        else:
            # Heuristic to decide first tool action (e.g., read_file on path, calc on expr)
            data = autowrap_to_action("", q)
            if "tool" in data:
                t = data["tool"]
                a = data.get("args", {})
                fn = TOOLS.get(t)
                if not fn:
                    return f"ERROR: unknown tool '{t}'"
                res = fn(a)
                return res if t == "read_file" else f"[TOOL RESULT] {res}"

    # Confident chat
    if route == "chat" and conf >= PLAN_CONF_THRESHOLD and not force_agent:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ]
        return llm(messages, temperature=0.3, max_tokens=600).strip()

    # --- FALLBACK: one normal call, then autowrap ---
    messages = [
        {"role": "system", "content": "You are an assistant. Respond naturally."},
        {"role": "user", "content": q},
    ]
    raw = llm(messages, temperature=0.3, max_tokens=500)
    data = autowrap_to_action(raw, q)

    if "tool" in data:
        t = data["tool"]
        a = data.get("args", {})
        fn = TOOLS.get(t)
        if not fn:
            return f"ERROR: unknown tool '{t}'"
        res = fn(a)
        return res if t == "read_file" else f"[TOOL RESULT] {res}"

    if "final" in data:
        return data["final"]

    return "ERROR: no valid result"

# -----------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------
if __name__ == "__main__":
    print("Local Mixtral agent (planner + sentinels + memory). Ctrl-C to quit.")
    while True:
        try:
            q = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        print("AI >", run_query(q))
