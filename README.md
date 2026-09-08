# Oromoo Version AI — Render + Telegram + Hugging Face

This package is a lightweight Telegram AI chatbot designed for a Render web service.

## What it does

- Telegram Bot API through an HTTPS webhook.
- Hugging Face Chat API with an open/open-weight model selected by `TEXT_MODEL`.
- Afaan Oromoo-first system prompt with multilingual replies.
- Live web research for current/latest/news questions.
- Direct URL scanning.
- **Hard cap: at most 5 scanned sources/pages for one user question.**
- Only 3 concurrent page fetch workers by default, to reduce Render Free CPU/RAM pressure.
- Known open/open-weight model knowledge questions skip web scanning unless the user asks for current/latest/news information.
- User conversation history is kept in memory.
- Simple per-user rate limiting and a global active-request limit protect free resources.
- `/health` and `/status` endpoints.
- Telegram webhook uses a secret token header.
- No polling loop is used, so there is no polling/webhook conflict.

## Files

- `app.py` — complete application.
- `requirements.txt` — Python packages.
- `render.yaml` — Render Blueprint.
- `.env.example` — example environment variables.
- `.gitignore` — keeps secrets out of GitHub.

## Render deployment

1. Create a GitHub repository.
2. Upload these files to the repository root.
3. In Render, create a new Web Service from the GitHub repository, or use the `render.yaml` Blueprint.
4. Choose the **Free** plan.
5. Add these two required environment variables:
   - `HF_TOKEN`
   - `TELEGRAM_BOT_TOKEN`
6. Deploy.

Render automatically provides `RENDER_EXTERNAL_URL`; the app uses it to configure the Telegram webhook after startup. You do not need to paste your Render URL into the source code.

## Important

Do **not** put `HF_TOKEN` or `TELEGRAM_BOT_TOKEN` in GitHub.

The Render Free web service sleeps after 15 minutes without inbound traffic. The next incoming request can wake it, and Render documents that the wake-up takes about one minute. This package therefore uses an HTTP webhook instead of Telegram polling.

The web scanner is intentionally conservative on Render Free:
- maximum 5 pages per question
- 3 parallel fetch workers
- short per-page timeouts
- response size cap
- no browser/Chromium/Playwright
- SSRF checks reject localhost/private/link-local targets

## Telegram commands

- `/start`
- `/help`
- `/web <question>`
- `/scan <url>`
- `/clear`
- `/status`

## Example

```text
Qwen 2.5 maal dha?
```

This is treated as a stable open-model knowledge question and does not trigger web scanning.

```text
/scan https://example.com
```

This scans that page and gives the model the extracted page content.

```text
/ web? 
```

Do not use a space after `/`. Use:

```text
/web oduu har'aa Ethiopia
```

## Hugging Face

`HF_PROVIDER=auto` lets the Hugging Face client choose an available inference provider for the selected model.

Change the model without editing the code:

```text
TEXT_MODEL=openai/gpt-oss-120b
```

For another compatible Hugging Face chat model, set `TEXT_MODEL` in Render Environment.

## Notes about persistence

User history, rate limits, metrics and the local HF guard are stored only in RAM. They reset when Render restarts, redeploys or spins down the free service. This is intentional to keep the free deployment simple and avoid needing a database.

## Official references

Render Free:
https://render.com/docs/free

Render web services:
https://render.com/docs/web-services

Render environment variables:
https://render.com/docs/environment-variables

Telegram Bot API:
https://core.telegram.org/bots/api

Hugging Face Inference Providers:
https://huggingface.co/docs/inference-providers/index
