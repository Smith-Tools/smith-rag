#!/usr/bin/env python3
"""Ollama embedding CLI wrapper using only stdlib (no external deps)"""
import sys
import json
import urllib.request

def embed(text: str, model: str = "mxbai-embed-large", base_url: str = "http://localhost:11434") -> list[float]:
    """Generate embedding for text using Ollama."""
    url = f"{base_url}/api/embed"
    data = json.dumps({"model": model, "input": text}).encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode('utf-8'))
        return result["embeddings"][0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ollama-embed.py <text>", file=sys.stderr)
        sys.exit(1)
    
    text = sys.argv[1]
    try:
        embedding = embed(text)
        print(json.dumps(embedding))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
