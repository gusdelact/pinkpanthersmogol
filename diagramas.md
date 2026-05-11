# Diagramas UML - Agentes RAG con smolagents

## Diagramas de Clases y Secuencia para `agente06.py` y `agente07.py`

---

## 1. Diagrama de Clases - agente06.py

```mermaid
classDiagram
    class Tool {
        <<abstract>>
        +String name
        +String description
        +Dict inputs
        +String output_type
        +forward()* 
    }

    class RAGRetrieverTool {
        +String name = "knowledge_base_search"
        +String description
        +Dict inputs
        +String output_type = "string"
        -Collection collection
        -SentenceTransformer embedding_model
        +__init__(collection, embedding_model)
        +forward(query: str) str
    }

    class CodeAgent {
        +List~Tool~ tools
        +LiteLLMModel model
        +int max_steps
        +int verbosity_level
        +run(query: str) str
    }

    class LiteLLMModel {
        +String model_id
        +String aws_region_name
    }

    class SentenceTransformer {
        +String model_name
        +encode(texts: list, show_progress_bar: bool) ndarray
    }

    class ChromaDB_Collection {
        +String name
        +Dict metadata
        +add(ids, documents, embeddings, metadatas)
        +query(query_embeddings, n_results) Dict
        +count() int
    }

    class ChromaDB_Client {
        +get_or_create_collection(name, metadata) Collection
    }

    Tool <|-- RAGRetrieverTool : hereda
    CodeAgent o-- LiteLLMModel : usa como modelo
    CodeAgent o-- RAGRetrieverTool : tiene como herramienta
    RAGRetrieverTool --> ChromaDB_Collection : consulta
    RAGRetrieverTool --> SentenceTransformer : genera embeddings
    ChromaDB_Client --> ChromaDB_Collection : crea/obtiene
```

---

## 2. Diagrama de Secuencia - agente06.py

```mermaid
sequenceDiagram
    participant U as Usuario
    participant M as main()
    participant LD as load_markdown_documents()
    participant CD as chunk_document()
    participant ST as SentenceTransformer
    participant CH as ChromaDB
    participant RT as RAGRetrieverTool
    participant CA as CodeAgent
    participant LLM as LiteLLMModel (Claude)

    Note over M: PASO 1 - Carga de documentos
    M->>LD: load_markdown_documents("kbpink/")
    LD-->>M: lista de documentos [{content, filename}]

    Note over M: PASO 2 - Construcción del vector store
    M->>CD: chunk_document(doc) por cada documento
    CD-->>M: lista de chunks [{content, chunk_id}]
    M->>ST: SentenceTransformer("all-MiniLM-L6-v2")
    ST-->>M: modelo cargado
    M->>ST: encode(textos)
    ST-->>M: embeddings (matriz numpy 384 dims)
    M->>CH: Client() → get_or_create_collection("pink_panther_kb")
    CH-->>M: collection
    M->>CH: collection.add(ids, documents, embeddings, metadatas)

    Note over M: PASO 3 - Crear herramienta y agente
    M->>RT: RAGRetrieverTool(collection, embedding_model)
    M->>CA: CodeAgent(tools=[retriever_tool], model, max_steps=4)

    Note over M: PASO 4 - Ejecutar consulta
    U->>M: query = "¿Qué reflexiones hace Pink Panther...?"
    M->>CA: agent.run(query)

    Note over CA,LLM: Loop de razonamiento (max 4 steps)
    CA->>LLM: Thought: "Necesito buscar en la KB..."
    LLM-->>CA: Code: knowledge_base_search(query="...")
    CA->>RT: forward("Pink Panther IA evolución humanidad")
    RT->>ST: encode([query])
    ST-->>RT: query_embedding
    RT->>CH: collection.query(query_embeddings, n_results=5)
    CH-->>RT: resultados (documentos + metadatas)
    RT-->>CA: "📚 Documentos recuperados: ..."
    CA->>LLM: Observation: documentos recuperados
    LLM-->>CA: Final Answer: respuesta sintetizada

    CA-->>M: resultado final
    M-->>U: 💡 Respuesta
```

---

## 3. Diagrama de Clases - agente07.py

```mermaid
classDiagram
    class Tool {
        <<abstract>>
        +String name
        +String description
        +Dict inputs
        +String output_type
        +forward()*
    }

    class RAGRetrieverTool {
        +String name = "knowledge_base_search"
        +String description
        +Dict inputs
        +String output_type = "string"
        -Collection collection
        -SentenceTransformer embedding_model
        +__init__(collection, embedding_model)
        +forward(query: str) str
    }

    class CodeAgent {
        +List~Tool~ tools
        +LiteLLMModel model
        +int max_steps
        +int verbosity_level
        +String instructions
        +run(query: str) str
    }

    class LiteLLMModel {
        +String model_id
        +String aws_region_name
    }

    class SentenceTransformer {
        +String model_name
        +encode(texts: list, show_progress_bar: bool) ndarray
    }

    class ChromaDB_Collection {
        +String name
        +Dict metadata
        +add(ids, documents, embeddings, metadatas)
        +query(query_embeddings, n_results) Dict
        +count() int
    }

    class ChromaDB_Client {
        +get_or_create_collection(name, metadata) Collection
    }

    class LiteLLM_Completion {
        <<module>>
        +completion(model, max_tokens, messages, aws_region_name) Response
    }

    class GradioBlocks {
        +String title
        +Tabs tabs
        +launch()
    }

    class GradioChatInterface {
        +Function fn
        +String description
        +List examples
    }

    Tool <|-- RAGRetrieverTool : hereda
    CodeAgent o-- LiteLLMModel : usa como modelo
    CodeAgent o-- RAGRetrieverTool : tiene como herramienta
    RAGRetrieverTool --> ChromaDB_Collection : consulta
    RAGRetrieverTool --> SentenceTransformer : genera embeddings
    ChromaDB_Client --> ChromaDB_Collection : crea/obtiene
    GradioBlocks *-- GradioChatInterface : contiene
    GradioChatInterface --> CodeAgent : invoca agent.run()
    LiteLLM_Completion ..> CodeAgent : genera instructions (personalidad)
```

---

## 4. Diagrama de Secuencia - agente07.py

```mermaid
sequenceDiagram
    participant U as Usuario (Browser)
    participant GR as Gradio UI
    participant GP as generate_personality()
    participant LIT as litellm.completion()
    participant LLM as Claude (Bedrock)
    participant LD as load_markdown_documents()
    participant CD as chunk_document()
    participant ST as SentenceTransformer
    participant CH as ChromaDB
    participant RT as RAGRetrieverTool
    participant CA as CodeAgent

    Note over GP,LLM: ═══ FASE 1: Pre-Agente (Generación de Personalidad) ═══

    GP->>GP: Lee kbpink/elprincipito.md (max 8000 chars)
    GP->>LIT: completion(model, messages=[system + user con texto Principito])
    LIT->>LLM: Prompt: "Genera personalidad Pink Panther + Principito"
    LLM-->>LIT: Personalidad generada (markdown, ~400 palabras)
    LIT-->>GP: response.choices[0].message.content
    GP-->>GP: GENERATED_PERSONALITY = personalidad

    Note over LD,CA: ═══ FASE 2: Construcción del Pipeline RAG ═══

    LD->>LD: load_markdown_documents("kbpink/")
    LD-->>LD: documentos cargados

    loop Por cada documento
        CD->>CD: chunk_document(doc, chunk_size=1000, overlap=200)
    end

    ST->>ST: SentenceTransformer("all-MiniLM-L6-v2")
    ST->>ST: encode(todos los textos)
    CH->>CH: Client() → get_or_create_collection("pink_panther_kb")
    CH->>CH: collection.add(ids, documents, embeddings, metadatas)

    RT->>RT: RAGRetrieverTool(collection, embedding_model)
    CA->>CA: CodeAgent(tools=[RT], model, instructions=GENERATED_PERSONALITY)

    Note over U,CA: ═══ FASE 3: Interacción del Usuario (Chat) ═══

    U->>GR: Escribe mensaje en el chat
    GR->>CA: chat(message) → agent.run(message)

    Note over CA,LLM: Loop de razonamiento con personalidad inyectada
    CA->>LLM: System prompt incluye GENERATED_PERSONALITY + Thought
    LLM-->>CA: Code: knowledge_base_search(query="...")
    CA->>RT: forward(query)
    RT->>ST: encode([query])
    ST-->>RT: query_embedding
    RT->>CH: collection.query(query_embeddings, n_results=5)
    CH-->>RT: documentos relevantes
    RT-->>CA: "📚 Documentos recuperados: ..."
    CA->>LLM: Observation + contexto recuperado
    LLM-->>CA: Final Answer (con personalidad Pink Panther + Principito)

    CA-->>GR: respuesta del agente
    GR-->>U: Muestra respuesta en el chat
```

---

## 5. Diagrama Comparativo de Arquitectura

```mermaid
flowchart TB
    subgraph agente06["agente06.py - RAG Básico"]
        direction TB
        A06_docs[📄 Documentos .md] --> A06_chunk[🔪 Chunking]
        A06_chunk --> A06_embed[📐 Embeddings]
        A06_embed --> A06_chroma[(ChromaDB)]
        A06_query[❓ Query hardcoded] --> A06_agent[🤖 CodeAgent]
        A06_agent --> A06_tool[🔍 RAGRetrieverTool]
        A06_tool --> A06_chroma
        A06_agent --> A06_llm[🧠 Claude via Bedrock]
        A06_llm --> A06_result[💡 Respuesta en consola]
    end

    subgraph agente07["agente07.py - RAG + Personalidad + Gradio"]
        direction TB
        A07_principito[📖 elprincipito.md] --> A07_preagent[🎭 Pre-Agente]
        A07_preagent --> A07_llm_pre[🧠 litellm.completion]
        A07_llm_pre --> A07_personality[✨ Personalidad Generada]

        A07_docs[📄 Documentos .md] --> A07_chunk[🔪 Chunking]
        A07_chunk --> A07_embed[📐 Embeddings]
        A07_embed --> A07_chroma[(ChromaDB)]

        A07_personality --> A07_agent[🤖 CodeAgent + instructions]
        A07_user[👤 Usuario via Gradio] --> A07_agent
        A07_agent --> A07_tool[🔍 RAGRetrieverTool]
        A07_tool --> A07_chroma
        A07_agent --> A07_llm[🧠 Claude via Bedrock]
        A07_llm --> A07_gradio[🖥️ Respuesta en Gradio UI]
    end
```

---

## 6. Resumen de Diferencias

| Aspecto | agente06.py | agente07.py |
|---------|-------------|-------------|
| **Personalidad** | Sin personalidad (respuestas neutras) | Generada dinámicamente por un pre-agente |
| **Interfaz** | Consola (ejecución única) | Gradio (chat web interactivo) |
| **Entrada** | Query hardcoded en el código | Usuario escribe en tiempo real |
| **Fases** | 1 fase (pipeline RAG directo) | 2 fases (pre-agente + RAG) |
| **Parámetro `instructions`** | No usado | Personalidad inyectada como system prompt |
| **Dependencias extra** | Ninguna | `gradio`, `litellm` (llamada directa) |
| **Patrón arquitectónico** | RAG simple | "Agente que configura a otro agente" |
| **Reutilización** | Ejecución única y termina | Servidor persistente, múltiples consultas |

---

*Documento generado como material de apoyo para la clase de Agentes con RAG.*
