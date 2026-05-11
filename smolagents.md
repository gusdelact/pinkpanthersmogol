# smolagents: Agentes Inteligentes con RAG

## Guía Completa con Ejemplos Prácticos

---

## 1. Introducción a smolagents

### 1.1 ¿Qué es smolagents?

smolagents es un framework ligero y open-source desarrollado por Hugging Face para construir agentes de inteligencia artificial. A diferencia de otros frameworks más pesados como LangChain o AutoGen, smolagents se enfoca en la simplicidad y la eficiencia: permite crear agentes funcionales con pocas líneas de código, sin sacrificar potencia.

Un agente, en el contexto de IA, es un programa que puede:
- Recibir una tarea en lenguaje natural
- Razonar sobre cómo resolverla
- Usar herramientas (tools) para obtener información o ejecutar acciones
- Iterar hasta encontrar una respuesta satisfactoria

smolagents implementa esto mediante el patrón "Thought-Action-Observation": el LLM piensa, ejecuta una acción (código o herramienta), observa el resultado, y repite hasta completar la tarea.

### 1.2 ¿Por qué smolagents?

Las ventajas principales de smolagents son:

- **Ligero**: Pocas dependencias, instalación rápida
- **CodeAgent**: Los agentes generan código Python ejecutable en lugar de JSON, lo que les da más flexibilidad
- **Agnóstico del modelo**: Funciona con cualquier LLM (OpenAI, Bedrock, Hugging Face, Ollama, etc.) gracias a LiteLLM
- **Herramientas personalizables**: Es fácil crear tools propias heredando de la clase `Tool`
- **Open-source**: Código abierto bajo licencia Apache 2.0

### 1.3 Arquitectura de un agente smolagents

```
┌─────────────────────────────────────────────────────┐
│                    CodeAgent                         │
├─────────────────────────────────────────────────────┤
│  LLM (modelo)          │  Tools (herramientas)      │
│  - Claude (Bedrock)    │  - knowledge_base_search   │
│  - GPT-4 (OpenAI)      │  - web_search              │
│  - Llama (local)       │  - calculator              │
│                        │  - custom_tool             │
├─────────────────────────────────────────────────────┤
│              Loop de razonamiento                    │
│  1. Thought (razonamiento)                          │
│  2. Code (genera código Python)                     │
│  3. Observation (resultado de ejecutar el código)   │
│  4. Repite hasta max_steps o final_answer           │
└─────────────────────────────────────────────────────┘
```

---

## 2. RAG: Retrieval-Augmented Generation

### 2.1 El problema que resuelve RAG

Los modelos de lenguaje (LLMs) tienen limitaciones fundamentales:
- No conocen información privada o interna de una organización
- Su conocimiento tiene una fecha de corte (no saben lo que pasó después)
- Pueden "alucinar" (inventar información que suena plausible pero es falsa)

RAG resuelve estos problemas conectando al LLM con una base de conocimiento externa. En lugar de depender solo de lo que el modelo "recuerda" de su entrenamiento, RAG le permite consultar documentos reales y basar sus respuestas en evidencia concreta.

### 2.2 Componentes de un sistema RAG

Un sistema RAG tiene tres componentes principales:

1. **Indexación**: Los documentos se procesan, se dividen en fragmentos (chunks), se convierten en vectores numéricos (embeddings) y se almacenan en una base de datos vectorial.

2. **Retrieval (Recuperación)**: Cuando llega una pregunta, se convierte en un vector y se buscan los fragmentos más similares en la base de datos.

3. **Generation (Generación)**: Los fragmentos recuperados se pasan al LLM como contexto, y el modelo genera una respuesta basada en esa información.

```
[Indexación]
Documentos → Chunking → Embeddings → Base de datos vectorial

[Consulta]
Pregunta → Embedding → Búsqueda por similitud → Top-K documentos
                                                       ↓
                                          LLM + Contexto → Respuesta
```

### 2.3 Tecnologías usadas en nuestros ejemplos

| Componente | Tecnología | Función |
|-----------|-----------|---------|
| Framework de agentes | smolagents | Orquestación del agente |
| LLM | Claude (Amazon Bedrock) | Razonamiento y generación |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Vectorización de texto |
| Base vectorial | ChromaDB | Almacenamiento y búsqueda de vectores |
| Interfaz | Gradio | UI web para chat |
| Gateway LLM | LiteLLM | Conexión unificada a múltiples proveedores |

---

## 3. Ejemplo 1: Agente RAG Básico (agente06.py)

### 3.1 Descripción general

Este es el ejemplo fundacional: un agente que puede responder preguntas sobre una knowledge base de historias de Pink Panther almacenadas en archivos markdown.

### 3.2 Paso 1: Carga de documentos

```python
import glob
import os

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kbpink")

def load_markdown_documents(directory: str) -> list[dict]:
    documents = []
    md_files = sorted(glob.glob(os.path.join(directory, "*.md")))
    for filepath in md_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        filename = os.path.basename(filepath)
        documents.append({
            "content": content,
            "filename": filename,
            "filepath": filepath,
        })
    return documents
```

La función recorre el directorio `kbpink/` y carga cada archivo `.md` como un diccionario con su contenido, nombre y ruta.

### 3.3 Paso 2: Chunking (fragmentación)

```python
def chunk_document(doc: dict, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    text = doc["content"]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        if end < len(text):
            last_newline = chunk_text.rfind("\n\n")
            if last_newline > chunk_size // 2:
                chunk_text = chunk_text[:last_newline]
                end = start + last_newline
        chunks.append({
            "content": chunk_text.strip(),
            "filename": doc["filename"],
            "chunk_id": f"{doc['filename']}_chunk_{len(chunks)}",
        })
        start = end - overlap if end < len(text) else len(text)
    return chunks
```

El chunking divide documentos largos en fragmentos de ~1000 caracteres con 200 caracteres de solapamiento (overlap). El overlap asegura que las ideas que caen en el borde entre dos chunks no se pierdan. El corte inteligente busca saltos de párrafo (`\n\n`) para no romper ideas a la mitad.

### 3.4 Paso 3: Embeddings y ChromaDB

```python
from sentence_transformers import SentenceTransformer
import chromadb

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, show_progress_bar=True)

client = chromadb.Client()
collection = client.get_or_create_collection(name="pink_panther_kb")
collection.add(
    ids=[chunk["chunk_id"] for chunk in all_chunks],
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[{"filename": chunk["filename"]} for chunk in all_chunks],
)
```

`SentenceTransformer("all-MiniLM-L6-v2")` carga un modelo que convierte texto en vectores de 384 dimensiones. Textos con significado similar producen vectores cercanos en el espacio vectorial. ChromaDB almacena estos vectores y permite buscar por similitud coseno.

### 3.5 Paso 4: Herramienta de Retrieval (smolagents Tool)

```python
from smolagents import Tool

class RAGRetrieverTool(Tool):
    name = "knowledge_base_search"
    description = (
        "Busca información relevante en la base de conocimiento de historias "
        "de Pink Panther."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "La consulta de búsqueda en lenguaje natural.",
        }
    }
    output_type = "string"

    def __init__(self, collection, embedding_model, **kwargs):
        super().__init__(**kwargs)
        self.collection = collection
        self.embedding_model = embedding_model

    def forward(self, query: str) -> str:
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=5
        )
        # Formatear y retornar resultados...
```

La clase `Tool` de smolagents requiere definir:
- `name`: Identificador que el agente usa para invocar la herramienta
- `description`: Texto que el LLM lee para decidir cuándo usarla (crucial para el comportamiento del agente)
- `inputs`: Esquema de parámetros que acepta
- `output_type`: Tipo de dato que retorna
- `forward()`: Método que ejecuta la lógica

### 3.6 Paso 5: Crear y ejecutar el agente

```python
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
    aws_region_name="us-east-1",
)

agent = CodeAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=4,
    verbosity_level=2,
)

result = agent.run("¿Qué piensa Pink Panther sobre la IA?")
```

El `CodeAgent` recibe la pregunta, genera código Python que llama a `knowledge_base_search()`, lee los resultados, y sintetiza una respuesta. `max_steps=4` limita las iteraciones para evitar loops infinitos.

---

## 4. Ejemplo 2: Agente con Personalidad Dinámica (agente07.py)

### 4.1 El concepto de "pre-agente"

El agente07 introduce un patrón avanzado: usar un LLM para configurar a otro LLM. Antes de crear el agente RAG, hacemos una llamada al modelo para que genere la personalidad del agente basándose en un texto externo (El Principito de Saint-Exupéry).

```
[Fase 1: Pre-agente]
elprincipito.md + conocimiento de Pink Panther → LLM → Personalidad generada

[Fase 2: Agente RAG]
Personalidad + Tools + Knowledge Base → Agente que responde con estilo
```

### 4.2 Generación dinámica de personalidad

```python
import litellm

def generate_personality() -> str:
    with open("kbpink/elprincipito.md", "r", encoding="utf-8") as f:
        principito_text = f.read()

    if len(principito_text) > 8000:
        principito_text = principito_text[:8000] + "\n\n[... texto continúa ...]"

    response = litellm.completion(
        model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens=500,
        messages=[
            {"role": "system", "content": "Eres un experto en diseño de personajes..."},
            {"role": "user", "content": f"Genera una personalidad que mezcle Pink Panther con El Principito basándote en: {principito_text}"},
        ],
        aws_region_name="us-east-1",
    )
    return response.choices[0].message.content
```

La llamada directa a `litellm.completion()` (sin agente) genera un prompt de personalidad que luego se inyecta en el agente RAG.

### 4.3 Inyección de personalidad con `instructions`

```python
GENERATED_PERSONALITY = generate_personality()

agent = CodeAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=4,
    verbosity_level=2,
    instructions=GENERATED_PERSONALITY,  # Personalidad dinámica
)
```

En smolagents 1.24.0, el parámetro `instructions` se inyecta dentro del system prompt base del agente. Esto permite que el agente mantenga su capacidad de usar herramientas y generar código, pero responda con la voz y filosofía del personaje generado.

### 4.4 Interfaz con Gradio (tabs)

```python
import gradio as gr

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Chat"):
            gr.ChatInterface(fn=chat, ...)
        with gr.Tab("Personalidad"):
            gr.Markdown(GENERATED_PERSONALITY)

demo.launch()
```

La interfaz usa tabs de Gradio: uno para chatear con el agente y otro para ver la personalidad que se generó dinámicamente.

---

## 5. Conceptos Clave en Profundidad

### 5.1 Embeddings y búsqueda semántica

Un embedding es una representación numérica del significado de un texto. El modelo `all-MiniLM-L6-v2` produce vectores de 384 dimensiones donde:

- "El gato duerme" y "El felino descansa" producen vectores MUY cercanos
- "El gato duerme" y "La economía crece" producen vectores MUY lejanos

La similitud coseno mide el ángulo entre dos vectores: 1.0 = idénticos, 0.0 = sin relación, -1.0 = opuestos.

### 5.2 ChromaDB como base vectorial

ChromaDB almacena documentos junto con sus embeddings y permite consultas por similitud. Cuando hacemos una query:
1. Se vectoriza la pregunta con el mismo modelo de embeddings
2. ChromaDB calcula la distancia coseno contra todos los vectores almacenados
3. Retorna los N documentos más cercanos (más relevantes semánticamente)

### 5.3 El loop del CodeAgent

```
Usuario: "¿Qué piensa Pink Panther sobre la música?"
    ↓
[Thought] "Necesito buscar información sobre Pink Panther y música"
    ↓
[Code] result = knowledge_base_search(query="Pink Panther reflexiones música")
    ↓
[Observation] "📚 Documentos recuperados: ..."
    ↓
[Thought] "Tengo suficiente información para responder"
    ↓
[Final Answer] "Pink Panther reflexiona sobre la música como..."
```

### 5.4 LiteLLM como gateway universal

LiteLLM permite usar la misma interfaz para conectarse a diferentes proveedores de LLM:

```python
# Amazon Bedrock
model = LiteLLMModel(model_id="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0")

# OpenAI
model = LiteLLMModel(model_id="gpt-4")

# Modelo local con Ollama
model = LiteLLMModel(model_id="ollama/llama3")
```

---

## 6. Despliegue y Producción

### 6.1 Google Colab

Para ejecutar en Colab, el dataset se descarga desde Kaggle:
```bash
!kaggle datasets download -d gustavodelacruztovar/historias-pink-panther
!unzip -o historias-pink-panther.zip -d kbpink
```

El token de Bedrock se configura via Colab Secrets o variable de entorno.

### 6.2 Hugging Face Spaces

Para desplegar como aplicación web permanente:
1. Crear un Space con SDK Gradio
2. Subir el código (`app.py`), dependencias (`requirements.txt`) y datos (`kbpink/`)
3. Configurar secretos: `AWS_BEARER_TOKEN_BEDROCK` y `HF_TOKEN`

El Space se construye automáticamente y queda disponible con una URL pública.

### 6.3 Estructura de archivos para Spaces

```
mi-space/
├── app.py              # Aplicación principal
├── requirements.txt    # smolagents, litellm, chromadb, sentence-transformers, gradio, boto3
├── README.md           # Metadata del Space (sdk: gradio, emoji, colores)
└── kbpink/             # Knowledge base
    ├── 01.md ... 13.md
    └── elprincipito.md
```

---

## 7. Conclusiones

smolagents ofrece un camino directo para construir agentes RAG funcionales sin la complejidad de frameworks más grandes. Los patrones demostrados en estos ejemplos son:

1. **RAG básico**: Documentos → Chunks → Embeddings → ChromaDB → Tool → CodeAgent
2. **Personalidad dinámica**: Pre-agente que configura al agente principal usando un LLM
3. **Interfaz interactiva**: Gradio para crear UIs de chat con mínimo código
4. **Despliegue**: Desde notebooks (Colab) hasta aplicaciones web (Hugging Face Spaces)

La combinación de smolagents + ChromaDB + sentence-transformers + LiteLLM proporciona un stack completo, ligero y flexible para construir aplicaciones de IA conversacional con acceso a conocimiento privado.

---

*Documento generado como material de apoyo para la clase de Agentes con RAG usando smolagents.*
*Repositorio: https://github.com/gusdelact/pinkpanthersmogol.git*
*Dataset: https://www.kaggle.com/datasets/gustavodelacruztovar/historias-pink-panther*
