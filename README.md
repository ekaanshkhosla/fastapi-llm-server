# Simple AI Server

## Overview

This project implements a lightweight **AI server** built with **FastAPI**, providing two main capabilities:
1. **Chat Completions** – A proxy endpoint compatible with OpenAI and OpenRouter models.
2. **Prefill** – Automatic extraction of invoice details from raw email text, saved into `data.csv`.

The solution demonstrates how to integrate LLM APIs into a simple service with structured outputs.

---

## Project Structure

```
├── emails/                # Sample email files for testing
├── data.csv               # Type of output we get
├── environment.yml        # Conda environment definition
├── main.py                # FastAPI server implementation
└── public_test.py         # Public test script
```

---

## Setup Instructions

### 1. Clone the Repository

### 2. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate ai-server
```

### 3. Add API Keys
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
```

---

## Running the Server

Start the server:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8090
```

Server will be available at:
```
http://localhost:8090
```

---

## API Endpoints

### 1. Chat Completions
**POST** `/v1/chat/completions`  

Example request:
```json
{
  "model": "moonshotai/kimi-k2:free",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Respond with Hi"}
  ],
  "max_tokens": 1000
}
```

---

### 2. Prefill
**POST** `/v1/prefill`  

Extracts and saves structured invoice fields (`amount`, `currency`, `due_date`, `description`, `company`, `contact`) to `data.csv`.

Example request:
```json
{
  "email_text": "Invoice #123: Amount $500 due by 2025-08-31. Contact billing@acme.com",
  "model": "moonshotai/kimi-k2:free"
}
```

Example response:
```json
{"success": true, "message": "data extracted and written"}
```

---

## Running Tests

Execute the public test script:
```bash
python public_test.py
```

This will:
- Call `/v1/chat/completions`
- Call `/v1/prefill`
- Display extracted CSV rows
- Clean up `data.csv`

