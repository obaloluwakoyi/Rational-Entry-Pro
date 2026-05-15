"""
ai_client.py — Unified AI client
Cloud: Anthropic, OpenAI, Gemini, Groq, Mistral, Together, Cohere
Local: Ollama, LM Studio, GPT4All, Jan, KoboldCpp, LocalAI
"""

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout


CLOUD_PROVIDERS = {
    "Anthropic (Claude)": {"default_model": "claude-sonnet-4-20250514", "key_hint": "sk-ant-…"},
    "OpenAI":             {"default_model": "gpt-4o-mini",               "key_hint": "sk-…"},
    "Google Gemini":      {"default_model": "gemini-1.5-flash",          "key_hint": "AIza…"},
    "Groq (free tier)":   {"default_model": "llama3-70b-8192",           "key_hint": "gsk_…"},
    "Mistral AI":         {"default_model": "mistral-small-latest",      "key_hint": "mistral key…"},
    "Together AI":        {"default_model": "meta-llama/Llama-3-70b-chat-hf", "key_hint": "together key…"},
    "Cohere":             {"default_model": "command-r-plus",            "key_hint": "cohere key…"},
}

LOCAL_PROVIDERS = {
    "Ollama":    {"url": "http://localhost:11434", "default_model": "llama3"},
    "LM Studio": {"url": "http://localhost:1234",  "default_model": "local-model"},
    "GPT4All":   {"url": "http://localhost:4891",  "default_model": "mistral-7b"},
    "Jan":       {"url": "http://localhost:1337",  "default_model": "llama3-8b"},
    "KoboldCpp": {"url": "http://localhost:5001",  "default_model": "kobold-model"},
    "LocalAI":   {"url": "http://localhost:8080",  "default_model": "gpt-3.5-turbo"},
}


class AIClient:
    def __init__(self, provider_type, provider_name, model,
                 api_key="", base_url="", timeout=120):
        self.provider_type = provider_type
        self.provider_name = provider_name
        self.model         = model
        self.api_key       = api_key
        self.base_url      = base_url.rstrip("/")
        self.timeout       = timeout

    def chat(self, messages: list[dict], max_tokens=1500, temperature=0.3) -> str:
        if self.provider_type == "local":
            return self._local(messages, max_tokens, temperature)
        return self._cloud(messages, max_tokens, temperature)

    def _local(self, messages, max_tokens, temperature) -> str:
        url  = self.base_url
        prov = self.provider_name
        endpoint = "/api/v1/generate" if prov == "KoboldCpp" else "/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if prov == "KoboldCpp":
            payload = {
                "prompt": "\n".join(m["content"] for m in messages),
                "max_length": max_tokens,
                "temperature": temperature,
            }

        try:
            r = requests.post(f"{url}{endpoint}", json=payload, timeout=self.timeout)
            r.raise_for_status()
        except RequestsConnectionError as exc:
            raise ConnectionError(
                f"Unable to connect to local model '{prov}' at {url}. "
                "Make sure the local server is running, the Base URL is correct, and the port is open."
            ) from exc
        except RequestsTimeout as exc:
            raise TimeoutError(
                f"Request to local model '{prov}' at {url} timed out. "
                "Try increasing the timeout or verify the server status."
            ) from exc

        if prov == "KoboldCpp":
            return r.json()["results"][0]["text"]
        return r.json()["choices"][0]["message"]["content"]

    def _cloud(self, messages, max_tokens, temperature) -> str:
        prov = self.provider_name
        key  = self.api_key
        if not key:
            raise ValueError(f"API key required for {prov}.")

        if prov == "Anthropic (Claude)":
            r = requests.post("https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json",
                         "x-api-key": key, "anthropic-version": "2023-06-01"},
                json={"model": self.model, "max_tokens": max_tokens, "messages": messages},
                timeout=self.timeout)
            r.raise_for_status()
            return "".join(b["text"] for b in r.json()["content"] if b.get("type") == "text")

        if prov == "Google Gemini":
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": m["content"]}]} for m in messages],
                      "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}},
                timeout=self.timeout)
            r.raise_for_status()
            return "".join(p["text"] for c in r.json().get("candidates", [])
                           for p in c.get("content", {}).get("parts", []) if "text" in p)

        if prov == "Cohere":
            r = requests.post("https://api.cohere.ai/v1/chat",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": self.model, "message": messages[-1]["content"],
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=self.timeout)
            r.raise_for_status()
            return r.json()["text"]

        urls = {
            "Groq (free tier)": "https://api.groq.com/openai/v1/chat/completions",
            "Mistral AI":       "https://api.mistral.ai/v1/chat/completions",
            "Together AI":      "https://api.together.xyz/v1/chat/completions",
            "OpenAI":           "https://api.openai.com/v1/chat/completions",
        }
        url = urls.get(prov, "https://api.openai.com/v1/chat/completions")
        r = requests.post(url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def build_analyst_prompt(results: list) -> str:
    lines = [
        f"{r.ticker} ({r.company_name}, {r.sector}): "
        f"Price ${r.price:.2f}, RSI {r.rsi:.1f}, SMA50 ${r.sma50:.2f}, "
        f"MA-Dev {r.ma_dev:+.1f}%, ATR ${r.atr:.2f}, "
        f"Stop-Loss ${r.stop_loss:.2f}, Risk Score {r.risk_score}/100 ({r.zone} zone)"
        for r in results
    ]
    return f"""You are a senior equity research analyst. Write a comprehensive professional report for these {len(results)} stock(s).

TECHNICAL DATA:
{chr(10).join(lines)}

Structure your report with these exact sections:

EXECUTIVE SUMMARY
2-3 sentences covering the overall picture across all tickers.

BEST ENTRY OPPORTUNITY
The best ticker to buy now, with specific entry price range, reasoning (RSI, SMA, momentum), and the exact stop-loss level.

HIGHEST RISK / AVOID
The most overextended ticker. Explain the technicals and what to watch.

COMPARATIVE ANALYSIS
Side-by-side comparison of all tickers: momentum strength, mean-reversion opportunity, volatility risk (ATR), and risk-adjusted profile.

STRATEGIC RECOMMENDATIONS
For each ticker: action (Buy / Hold / Avoid), entry condition, stop-loss, and rationale in one paragraph.

RISK DISCLAIMER
One sentence.

Style: specific with price levels, data-driven, professional sell-side analyst tone. Bold ticker symbols using **TICKER**.
"""


def build_chat_system(results: list) -> str:
    if not results:
        return "You are a helpful stock analyst. No data loaded yet — ask the user to run the scanner first."
    lines = [
        f"- **{r.ticker}** ({r.company_name}, {r.sector}): "
        f"Price ${r.price:.2f}, RSI {r.rsi:.1f}, SMA50 ${r.sma50:.2f} ({r.ma_dev:+.1f}% deviation), "
        f"ATR ${r.atr:.2f}, Stop-Loss ${r.stop_loss:.2f}, "
        f"Risk Score {r.risk_score}/100 ({r.zone} zone), Last: {r.last_date}"
        for r in results
    ]
    return (
        "You are an expert stock analyst assistant with live technical analysis data:\n\n"
        + "\n".join(lines)
        + "\n\nAnswer questions based on this data. Be concise, specific, cite exact numbers. "
          "If asked something outside this data, say so clearly. Never fabricate data."
    )
