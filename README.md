---
title: Pink Panther Agent
emoji: 🐾
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "5.31.0"
app_file: app.py
pinned: false
---

# 🌹 Pink Panther como El Principito - Chat RAG

Agente RAG con personalidad generada dinámicamente que mezcla la elegancia de la Pink Panther con la filosofía del Principito de Saint-Exupéry.

## Arquitectura

1. **Pre-agente**: Lee El Principito y genera una personalidad híbrida usando el LLM
2. **Agente RAG**: Usa esa personalidad para responder preguntas sobre la knowledge base de historias de Pink Panther

## Tecnologías

- smolagents (Hugging Face)
- ChromaDB (base de datos vectorial)
- sentence-transformers (embeddings)
- Amazon Bedrock / Claude (LLM)
- Gradio (interfaz web)

## Secretos requeridos

Configura en Settings → Repository secrets:

- `AWS_BEARER_TOKEN_BEDROCK`: Token de acceso a Amazon Bedrock
