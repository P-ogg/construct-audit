"""One place that talks to language models. Two interchangeable backends:

  AnthropicClient     Claude through the Anthropic SDK (structured JSON output, refusal fallback)
  OpenAICompatClient  any OpenAI-compatible chat server: open-weights models on OpenRouter, Groq, Together,
                      or locally in LM Studio, Ollama, llama.cpp, vLLM

Both expose .json(system, user, schema) -> dict and a .usage record, so every stage works with either.
Keys come from arguments or the environment; nothing here stores or prints them.

    make_client("anthropic")                                                 # ANTHROPIC_API_KEY
    make_client("openai-compat", model="qwen/qwen3-235b-a22b",
                base_url="https://openrouter.ai/api/v1")                     # OPENROUTER_API_KEY / OPENAI_API_KEY
    make_client("openai-compat", model="qwen3.5-2b", base_url="http://localhost:1234/v1")   # LM Studio, no key
"""
import json, os, re, threading, time
from dataclasses import dataclass, field

import httpx
import jsonschema

ANTHROPIC_MODEL = "claude-opus-5-5"
PRICE = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}  # $ per 1M in/out
MODEL = ANTHROPIC_MODEL


_LOCK = threading.Lock()


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    log: list = field(default_factory=list)
    price: tuple = None              # ($ per 1M input, $ per 1M output), if known
    waits: int = 0                   # times the provider's rate limit made us wait
    waited: float = 0.0              # seconds spent waiting for it

    def note_wait(self, seconds):
        with _LOCK:
            self.waits += 1; self.waited += seconds

    def add(self, label, inp, out, stop, rid=""):
        with _LOCK:                      # several articles may be read at once
            self.calls += 1; self.input_tokens += inp or 0; self.output_tokens += out or 0
            self.log.append(dict(label=label, input=inp, output=out, stop=stop, id=rid))

    def cost(self, model=None):
        p = self.price or PRICE.get(model or "", None)
        return None if p is None else (self.input_tokens * p[0] + self.output_tokens * p[1]) / 1e6


def parse_json(text):
    """JSON from a model reply: strips <think> blocks and code fences, then takes the first balanced object."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth, instr, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if instr:
                esc = (ch == "\\" and not esc)
                if ch == '"' and not esc:
                    instr = False
                continue
            if ch == '"':
                instr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object in reply")


class AnthropicClient:
    def __init__(self, api_key=None, model=ANTHROPIC_MODEL, effort="high", use_env=True):
        import anthropic
        self._anthropic = anthropic
        if not api_key and not use_env:
            raise ValueError("no API key was given")
        # retries are handled below (max_retries=0), so rate-limit waits can be counted and shown to the user
        self.c = anthropic.Anthropic(api_key=api_key, max_retries=0) if api_key else anthropic.Anthropic(max_retries=0)
        self.model, self.effort = model, effort
        self.usage = Usage(price=PRICE.get(model))

    native_pdf = True

    def _content(self, user, doc, images, doc_first):
        if doc is None and not images:
            return user
        parts = []
        if doc is not None:
            if doc.get("pdf_b64"):
                parts.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": doc["pdf_b64"]}})
            else:
                parts.append({"type": "text", "text": "ARTICLE TEXT\n" + doc["text"]})
        for b64, mime in images or []:
            parts.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
        instr = {"type": "text", "text": user}
        return parts + [instr] if doc_first else [instr] + parts

    def json(self, system, user, schema, max_tokens=32000, label="", doc=None, images=None, doc_first=False):
        a = self._anthropic
        for attempt in range(7):
            try:
                with self.c.beta.messages.stream(
                    model=self.model, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": self._content(user, doc, images, doc_first)}],
                    output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                ) as s:
                    r = s.get_final_message()
                break
            except (a.RateLimitError, a.InternalServerError, a.APIConnectionError) as e:
                if attempt == 6:
                    raise
                wait = 15.0 * (attempt + 1)
                if isinstance(e, a.RateLimitError):
                    try:
                        wait = max(5.0, float(e.response.headers.get("retry-after", wait)))
                    except (TypeError, ValueError, AttributeError):
                        pass
                    self.usage.note_wait(wait)
                time.sleep(wait)
        self.usage.add(label, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason, getattr(r, "id", ""))
        if r.stop_reason == "refusal":
            raise RuntimeError(f"model declined ({label})")
        if r.stop_reason == "max_tokens":
            raise RuntimeError(f"output cut off at max_tokens ({label})")
        return json.loads(next(b.text for b in r.content if b.type == "text"))


class OpenAICompatClient:
    """OpenAI-compatible /chat/completions. Asks for schema-constrained JSON; if the server or model cannot do that,
    falls back to prompting for JSON. Every reply is parsed, validated against the schema, and retried with the
    validation error shown to the model (up to `retries` times)."""

    def __init__(self, base_url, model, api_key=None, price=None, extra=None, structured=True, retries=2, timeout=900, vision=False,
                 max_output=None, use_env=True):
        self.max_output = max_output         # local servers have small context windows: cap the reply length
        self.vision = vision                 # send table images only to models that can see
        self.base = base_url.rstrip("/")
        self.model, self.extra, self.structured, self.retries, self.timeout = model, extra or {}, structured, retries, timeout
        # a hosted app passes use_env=False: a visitor must never fall back on a key the server happens to have
        key = api_key or (use_env and (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")))
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = f"Bearer {key}"
        self.usage = Usage(price=price)

    def _post(self, body):
        for attempt in range(4):
            r = httpx.post(f"{self.base}/chat/completions", json=body, headers=self.headers, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                wait = 10.0 * (attempt + 1)
                if r.status_code == 429:
                    try:
                        wait = max(5.0, float(r.headers.get("retry-after", wait)))
                    except (TypeError, ValueError):
                        pass
                    self.usage.note_wait(wait)
                time.sleep(wait); continue
            return r

    native_pdf = False

    def _content(self, user, doc, images, doc_first):
        if doc is None and not images:
            return user
        parts = []
        if doc is not None:
            parts.append({"type": "text", "text": "ARTICLE TEXT\n" + doc["text"]})
        if self.vision:
            for b64, mime in images or []:
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        instr = {"type": "text", "text": user}
        return parts + [instr] if doc_first else [instr] + parts

    def json(self, system, user, schema, max_tokens=32000, label="", doc=None, images=None, doc_first=False):
        sys_prompt = system if self.structured else (
            system + "\n\nReply with one JSON object only, no other text, matching this JSON Schema:\n" + json.dumps(schema))
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": self._content(user, doc, images, doc_first)}]
        last = None
        for attempt in range(self.retries + 1):
            body = {"model": self.model, "messages": msgs, "max_tokens": min(max_tokens, self.max_output or max_tokens), "temperature": 0, **self.extra}
            if self.structured:
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}}
            r = self._post(body)
            if r.status_code == 400 and self.structured:          # server cannot constrain output: prompt for JSON instead
                self.structured = False
                return self.json(system, user, schema, max_tokens, label, doc, images, doc_first)
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]; u = data.get("usage") or {}
            self.usage.add(label, u.get("prompt_tokens"), u.get("completion_tokens"), choice.get("finish_reason"), data.get("id", ""))
            text = choice["message"].get("content") or ""
            if not text.strip() and choice.get("finish_reason") == "length":
                raise RuntimeError(f"the model used all {max_tokens} tokens thinking and gave no answer ({label}); "
                                   "turn thinking off for this provider (extra=...) or raise max_tokens")
            try:
                obj = parse_json(text)
                jsonschema.validate(obj, schema)
                return obj
            except (ValueError, jsonschema.ValidationError) as e:
                last = e
                msgs += [{"role": "assistant", "content": text[:4000]},
                         {"role": "user", "content": f"That reply was not valid: {str(e)[:500]}. Reply again with the JSON object only."}]
        raise RuntimeError(f"no valid JSON after {self.retries + 1} tries ({label}): {last}")


def make_client(provider="anthropic", model=None, base_url=None, api_key=None, **kw):
    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model or ANTHROPIC_MODEL, **{k: v for k, v in kw.items() if k in ("effort", "use_env")})
    if provider in ("openai-compat", "openai", "openrouter", "groq", "lmstudio", "ollama"):
        default = {"openrouter": "https://openrouter.ai/api/v1", "groq": "https://api.groq.com/openai/v1",
                   "lmstudio": "http://localhost:1234/v1", "ollama": "http://localhost:11434/v1"}
        url = base_url or default.get(provider)
        if not url or not model:
            raise ValueError("openai-compat needs base_url and model")
        # Thinking models spend the whole token budget reasoning unless told not to; the switch is named differently per server.
        no_think = {"lmstudio": {"reasoning_effort": "none"}, "ollama": {"reasoning_effort": "none"},
                    "groq": {"reasoning_effort": "none"}, "openrouter": {"reasoning": {"enabled": False}}}
        kw.setdefault("extra", no_think.get(provider, {}))
        return OpenAICompatClient(url, model, api_key=api_key, **{k: v for k, v in kw.items() if k != "effort"})
    raise ValueError(f"unknown provider {provider}")


Client = AnthropicClient   # backwards compatibility


def explain(err):
    """A model or connection error in one plain sentence, for people who are not programmers."""
    t = f"{type(err).__name__}: {err}".lower()
    if "no api key" in t:
        return "No API key was entered. Add one in step 2."
    if any(x in t for x in ("401", "authentication", "invalid x-api-key", "invalid api key", "unauthorized", "incorrect api key")):
        return "The API key was refused. Check that you copied the whole key, and that it belongs to the service you chose."
    if any(x in t for x in ("402", "credit", "insufficient", "billing", "quota")):
        return "The service refused because the account has no credit left. Add credit on the provider's website."
    if any(x in t for x in ("404", "not_found", "model not found", "does not exist", "no such model", "is not a valid model")):
        return "The service does not know this model name. Check the spelling on the provider's model list."
    if "429" in t or "rate limit" in t:
        return "The service is limiting how fast this key can send requests. Wait a few minutes and press Start again; finished articles are kept."
    if any(x in t for x in ("connect", "timed out", "timeout", "name or service not known", "nodename", "connection refused")):
        return "The model's server could not be reached. Check the service and your internet connection."
    if any(x in t for x in ("cut off at max_tokens", "tokens thinking")):
        return "The model ran out of room for its answer. Try a larger model."
    if "no valid json" in t:
        return "The model did not answer in the required format, even after being asked again. Try a larger model."
    if "declined" in t or "refusal" in t:
        return "The model declined to read this article."
    return f"Something went wrong: {str(err)[:200]}"
