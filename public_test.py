"""Simple public test for candidate to run"""

import os
import csv
import json
import requests

SERVER_URL = "http://localhost:8090"
MODEL = "moonshotai/kimi-k2:free"      # specify you model

def test_chat_completions():
    """Test OpenAI proxy endpoint"""
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,        # Your Model Name
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Respond with Hi"}
        ],
        "max_tokens": 1000,
    }

    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"].strip()
    print(f"✓ Chat completions: {content}")




def test_prefill_simple():
    """Test prefill endpoint with all emails in the emails/ directory"""
    url = f"{SERVER_URL}/v1/prefill"

    # Loop through all email files
    for filename in os.listdir("emails"):
        if filename.endswith(".txt"):
            file_path = os.path.join("emails", filename)
            with open(file_path, "r", encoding="utf-8") as f:
                email_text = f.read()

            payload = {
                "email_text": email_text,
                "model": MODEL
            }

            response = requests.post(url, json=payload)
            assert response.status_code == 200
            data = response.json()
            print(f"✓ Prefill from {filename}: {json.dumps(data, separators=(',', ':'))}")
            assert data["success"] is True



    if os.path.exists("data.csv"):
        with open("data.csv", "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                print(f"Row {i + 1}: {json.dumps(dict(row), separators=(',', ':'))}")


def cleanup_csv():
    """Clean up CSV file after tests"""
    import os

    if os.path.exists("data.csv"):
        os.remove("data.csv")
        print("Cleaned up data.csv file")


if __name__ == "__main__":
    try:
        test_chat_completions()
        test_prefill_simple()
        print("All tests passed!")
    finally:
        print("DONE")      # Comment it if do not want to save data and uncommnet next line for cleanup
        # cleanup_csv()
