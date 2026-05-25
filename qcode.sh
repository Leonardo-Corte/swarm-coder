#!/bin/bash
# qcode - qwen-code pointed at local Ollama
export OLLAMA_API_KEY=ollama
exec qwen --auth-type openai "$@"
