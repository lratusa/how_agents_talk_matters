"""DeepSeek chat client (async httpx, concurrency-limited, retrying) and
Ollama bge-m3 embedder with a sha256-keyed JSONL disk cache.

Also provides deterministic mock implementations for offline smoke tests.
"""
import asyncio
import hashlib
import json
import os
import random
import time

import httpx
import numpy as np

from . import config


def load_api_key(path=config.ENV_KEY_FILE, name=config.ENV_KEY_NAME):
    """Read `name=value` from a dotenv file with a UTF-8 BOM."""
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("%s not found in %s" % (name, path))


class ChatClient:
    """Async chat client: semaphore-bounded concurrency, exponential backoff
    on 429/5xx/timeouts, per-call usage + latency capture."""

    def __init__(self, model=None, max_concurrency=16, api_key=None,
                 max_retries=12, timeout=120.0):
        self.model = model or config.MODEL
        self.api_key = api_key or load_api_key()
        self.max_retries = max_retries
        self.timeout = timeout
        self._max_conc = max_concurrency
        # Loop-local primitives: the pipeline runs stages in separate
        # asyncio.run() loops, so semaphores/httpx clients must be created
        # per running loop (a Semaphore created outside a loop binds to the
        # first loop and breaks subsequent ones).
        self._sems = {}
        self._clients = {}
        self.calls_made = 0  # diagnostic counter (used by smoke tests)

    def _sem(self):
        loop = asyncio.get_running_loop()
        if loop not in self._sems:
            self._sems[loop] = asyncio.Semaphore(self._max_conc)
        return self._sems[loop]

    async def _session(self):
        loop = asyncio.get_running_loop()
        if loop not in self._clients:
            self._clients[loop] = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Authorization": "Bearer %s" % self.api_key},
            )
        return self._clients[loop]

    async def close(self):
        # Best-effort cleanup: clients/semaphores belong to their own
        # (possibly already closed) loops; failures here are harmless noise
        # on Windows proactor loops, so swallow them.
        for client in self._clients.values():
            try:
                await client.aclose()
            except Exception:
                pass
        self._clients = {}
        self._sems = {}

    async def chat(self, prompt, max_tokens):
        """Single user-message call. Returns dict(content, model, usage, latency_s)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "max_tokens": max_tokens,
            # no seed: provider does not support it (prereg §1)
        }
        if config.THINKING_DISABLED:
            payload["thinking"] = {"type": "disabled"}  # Amendment 6
        payload.update(config.EXTRA_PAYLOAD)
        async with self._sem():
            client = await self._session()
            last_err = None
            for attempt in range(self.max_retries):
                t0 = time.monotonic()
                try:
                    resp = await client.post(config.API_BASE, json=payload)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            "retryable status %s" % resp.status_code,
                            request=resp.request, response=resp)
                    resp.raise_for_status()
                    data = resp.json()
                    latency = time.monotonic() - t0
                    self.calls_made += 1
                    msg = data["choices"][0]["message"]["content"]
                    usage = data.get("usage") or {}
                    return {
                        "content": msg,
                        # exact served model name as returned by the API (prereg §1)
                        "model": data.get("model", self.model),
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "total_tokens": usage.get("total_tokens"),
                        },
                        "latency_s": round(latency, 4),
                    }
                except (httpx.HTTPStatusError, httpx.TimeoutException,
                        httpx.TransportError) as e:
                    last_err = e
                    retryable = True
                    if isinstance(e, httpx.HTTPStatusError):
                        code = e.response.status_code
                        retryable = code == 429 or code >= 500
                    if not retryable or attempt == self.max_retries - 1:
                        raise
                    sleep = min(4.0 * (2.0 ** attempt) + random.random() * 5,
                                180.0)
                    await asyncio.sleep(sleep)
            raise last_err  # unreachable


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Embedder:
    """bge-m3 via Ollama's OpenAI-compatible endpoint, with a JSONL disk cache
    keyed by sha256(text). Embeddings are cached forever (text-addressed)."""

    def __init__(self, cache_path=config.EMBED_CACHE, batch_size=16):
        self.cache_path = cache_path
        self.model_name = config.EMBED_MODEL
        self.batch_size = batch_size
        self._cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["embedding"]

    def _embed_batch(self, texts):
        resp = httpx.post(
            config.OLLAMA_BASE + "/v1/embeddings",
            json={"model": config.EMBED_MODEL, "input": texts},
            timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def embed_texts(self, texts):
        """Returns list of vectors aligned with `texts` (dedupes via cache)."""
        missing = [t for t in dict.fromkeys(texts) if _sha256(t) not in self._cache]
        new_rows = []
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            for text, vec in zip(batch, self._embed_batch(batch)):
                row = {"key": _sha256(text), "model": config.EMBED_MODEL,
                       "embedding": vec}
                self._cache[row["key"]] = vec
                new_rows.append(row)
        if new_rows:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                for row in new_rows:
                    f.write(json.dumps(row) + "\n")
        return [self._cache[_sha256(t)] for t in texts]


# ---------------------------------------------------------------------------
# Deterministic mocks (offline smoke tests / --mock runs). No network.
# ---------------------------------------------------------------------------

class MockChatClient:
    """Deterministic fake LLM. Output depends only on sha256(prompt), so the
    whole mock run is reproducible. Understands the three frozen prompt shapes
    by the end-marker each one requests."""

    def __init__(self, model="mock-model", max_concurrency=None, **kw):
        self.model = model
        self.calls_made = 0

    async def close(self):
        pass

    @staticmethod
    def _token(prompt, salt):
        h = hashlib.sha256((salt + "|" + prompt).encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    async def chat(self, prompt, max_tokens):
        self.calls_made += 1
        await asyncio.sleep(0)  # keep the async contract
        # detect by the FINAL requested marker (synth prompts may embed review
        # texts that contain the words "REVIEW VERDICT:")
        tail = prompt.rstrip()
        is_review = tail.endswith("YOUR UPDATED ANSWER: <letter>") or \
            tail.endswith("YOUR UPDATED ANSWER: <value>")
        if is_review:
            verdict = "PEER CORRECT" if self._token(prompt, "v") % 2 == 0 else "PEER INCORRECT"
            letter = chr(ord("A") + self._token(prompt, "u") % 4)
            content = (
                "1. The setup is reasonable.\n2. Possible arithmetic slips.\n"
                "3. Assumptions need checking.\n4. See verdict.\n5. See below.\n"
                "REVIEW VERDICT: %s\nYOUR UPDATED ANSWER: %s" % (verdict, letter))
        elif "FINAL ANSWER: <letter>" in prompt:
            letter = chr(ord("A") + self._token(prompt, "a") % 4)
            content = ("Step by step reasoning (mock %d).\nJustification: mock.\n"
                       "FINAL ANSWER: %s" % (self._token(prompt, "j") % 1000, letter))
        else:  # free-answer variant: FINAL ANSWER: <value>
            value = str(self._token(prompt, "a") % 100)
            content = ("Step by step reasoning (mock).\nJustification: mock.\n"
                       "FINAL ANSWER: %s" % value)
        n_prompt = len(prompt) // 4
        n_completion = len(content) // 4
        return {
            "content": content,
            "model": self.model,
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_completion,
                      "total_tokens": n_prompt + n_completion},
            "latency_s": 0.0,
        }


class MockEmbedder:
    """Deterministic pseudo-embeddings: seeded numpy RNG from sha256(text).
    64-dim (NOT 1024) — mock only, never mixed with real cache files."""

    dim = 64
    model_name = "mock-embedder"

    def __init__(self, cache_path=None, **kw):
        self.cache_path = cache_path  # accepted for interface parity; unused

    def embed_texts(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            out.append(rng.randn(self.dim).tolist())
        return out


def make_clients(mock, model=None, max_concurrency=16, embed_cache=None):
    """Factory: returns (chat_client, embedder)."""
    if mock:
        return MockChatClient(), MockEmbedder()
    return (ChatClient(model=model, max_concurrency=max_concurrency),
            Embedder(cache_path=embed_cache or config.EMBED_CACHE))
