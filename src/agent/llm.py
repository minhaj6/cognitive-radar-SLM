"""Thin wrapper around the Ollama Python client.

We can swap LLM model in Ollama as long as replacement model exposes a 
`chat(messages, tools=...)` method that returns an object with `.message.content`
 and `.message.tool_calls`, the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ollama import Client


@dataclass
class LLMAgent:
    model: str
    host: str = "http://localhost:11434"
    options: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    tool_specs: list[dict] = field(default_factory=list)
    tool_registry: dict[str, Callable[..., Any]] = field(default_factory=dict)
    _client: Client = field(init=False)

    def __post_init__(self) -> None:
        self._client = Client(host=self.host)

    def ping(self) -> bool:
        """Verify the Ollama host is reachable and the model is pulled."""
        try:
            self._client.show(self.model)
            return True
        except Exception:
            return False

    def run(
        self,
        user_query: str,
        *,
        max_steps: int = 8,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> list[dict]:
        """
        ``on_event(kind, payload)``
        
        Event kinds: ``step``, ``assistant_start``, ``token``, ``assistant_end``, 
                     ``tool_call``, ``tool_result``.

        Returns the full transcript (list of message dicts) for logging.
        """

        def emit(kind: str, payload: dict | None = None) -> None:
            if on_event is not None:
                on_event(kind, payload or {})

        transcript: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        empty_retries = 0
        MAX_EMPTY_RETRIES = 1

        for step in range(max_steps):
            emit("step", {"index": step})
            content, tool_calls = self._stream_chat(transcript, emit)

            if not tool_calls and not content.strip():
                if empty_retries < MAX_EMPTY_RETRIES:
                    empty_retries += 1
                    nudge = (
                        "Your previous turn was empty. The conversation cannot end "
                        "without a non-empty summary. Produce the final summary "
                        "now: restate the user's question in one sentence, quote "
                        "the concrete numbers from the most recent tool results "
                        "(estimated DOAs in degrees, SLL in dB, beamwidth, null "
                        "depth, output SINR -- whichever apply), and name any "
                        "saved plot files verbatim. Plain prose, no JSON, no tool "
                        "calls."
                    )
                    transcript.append({"role": "user", "content": nudge})
                    emit("nudge", {"reason": "empty_terminal_turn"})
                    continue
                # Already retried -- record the empty turn and give up.
                transcript.append({"role": "assistant", "content": content, "tool_calls": []})
                break

            empty_retries = 0
            transcript.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [_serialize_tool_call(tc) for tc in tool_calls],
            })

            if not tool_calls:
                break

            for tc in tool_calls:
                name, args = _extract_call(tc)
                emit("tool_call", {"name": name, "arguments": args})
                handler = self.tool_registry.get(name)
                if handler is None:
                    observation = {"error": f"tool '{name}' is not registered"}
                else:
                    try:
                        observation = handler(**args)
                    except Exception as exc:  # surface errors back to the model
                        observation = {"error": f"{type(exc).__name__}: {exc}"}
                emit("tool_result", {"name": name, "result": observation})

                transcript.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(observation, default=_json_default),
                })

        return transcript

    def _stream_chat(
        self,
        transcript: list[dict],
        emit: Callable[[str, dict], None],
    ) -> tuple[str, list]:
        """Stream one chat turn; emit token events as they arrive. Returns the
        accumulated assistant content and any tool_calls from the final chunk."""
        accumulated = ""
        tool_calls: list = []
        emit("assistant_start", {})
        try:
            stream = self._client.chat(
                model=self.model,
                messages=transcript,
                tools=self.tool_specs or None,
                options=self.options or None,
                stream=True,
            )
        except TypeError:
            # Fallback: this client doesn't accept stream=True.
            resp = self._client.chat(
                model=self.model,
                messages=transcript,
                tools=self.tool_specs or None,
                options=self.options or None,
            )
            msg = resp["message"] if isinstance(resp, dict) else resp.message
            accumulated = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
            if accumulated:
                emit("token", {"text": accumulated})
            tcs = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
            tool_calls = list(tcs or [])
            emit("assistant_end", {"content": accumulated, "has_tool_calls": bool(tool_calls)})
            return accumulated, tool_calls

        for chunk in stream:
            msg = chunk["message"] if isinstance(chunk, dict) else chunk.message
            delta = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
            if delta:
                accumulated += delta
                emit("token", {"text": delta})
            chunk_tcs = (
                msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
            )
            if chunk_tcs:
                tool_calls.extend(chunk_tcs)
        emit("assistant_end", {"content": accumulated, "has_tool_calls": bool(tool_calls)})
        return accumulated, tool_calls


def _extract_call(tc: Any) -> tuple[str, dict]:
    fn = tc["function"] if isinstance(tc, dict) else tc.function
    name = fn["name"] if isinstance(fn, dict) else fn.name
    raw_args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
    if isinstance(raw_args, str):
        args = json.loads(raw_args) if raw_args else {}
    else:
        args = dict(raw_args or {})
    return name, args


def _serialize_tool_call(tc: Any) -> dict:
    name, args = _extract_call(tc)
    return {"function": {"name": name, "arguments": args}}


def _json_default(obj: Any):
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except ImportError:
        pass
    return str(obj)
