# PA Chat Bot

FastAPI service that chats over notes using gemini-app and pinecone-app.

## Endpoint

- `POST /summarize`

### Request body

```json
{
  "text": "Long input text...",
  "max_words": 120
}
```

### Response body

```json
{
  "summary": "Short summary...",
  "model": "gemini-1.5-flash",
  "input_char_count": 12345
}
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables (use one based on environment):

```bash
cp .env.dev .env
```

Then set `GOOGLE_API_KEY` in `.env`.
Also set `DATABASE_URL` for your Supabase Postgres connection.

Available environment files:
- `.env.dev`
- `.env.test`
- `.env.prod`

To run with a specific environment file:

```bash
APP_ENV=dev uvicorn app.main:app --reload
```

You can replace `dev` with `test` or `prod`.

Supabase connection string tip:
- URL-encode special characters in the password (for example `@` becomes `%40`).
- Example format: `postgresql://postgres:<encoded_password>@db.<project-ref>.supabase.co:5432/postgres`

## Run

```bash
uvicorn app.main:app --reload
```

Server starts at `http://127.0.0.1:8000`.

## Test

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Paste long text here.",
    "max_words": 100
  }'
```

Open API docs at `http://127.0.0.1:8000/docs`.
