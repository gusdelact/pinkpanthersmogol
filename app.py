"""
Agente 07 - RAG con personalidad generada dinámicamente (Gradio)
=================================================================
Versión del agente RAG donde la personalidad de Pink Panther se genera
dinámicamente usando un "pre-agente": una llamada previa al LLM que lee
el libro de El Principito (kbpink/elprincipito.md) y, combinándolo con
su conocimiento público de la Pink Panther, genera un system prompt de
personalidad único cada vez que se ejecuta.

Arquitectura de dos fases:
    [Fase 1 - Pre-agente]
    Lee elprincipito.md → LLM genera personalidad combinada Pink Panther + Principito

    [Fase 2 - Agente RAG]  
    Usa la personalidad generada como system_prompt → Responde preguntas con RAG

Usa:
- smolagents + ChromaDB + sentence-transformers: Para el pipeline RAG
- LiteLLM + Bedrock (Claude): Como modelo de lenguaje
- Gradio: Para la interfaz de chat en el navegador

Ejecutar:
    python agente07.py

Se abrirá una interfaz web en http://localhost:7860
"""

import os
import glob
import chromadb
import gradio as gr
from sentence_transformers import SentenceTransformer
from smolagents import Tool, CodeAgent, LiteLLMModel
import litellm

# ==============================================================================
# Configuración del modelo (Bedrock - Claude)
# ==============================================================================
os.environ["AWS_REGION_NAME"] = "us-east-1"
# AWS_BEARER_TOKEN_BEDROCK se lee de la variable de entorno ya configurada.
# Exportala antes de ejecutar: export AWS_BEARER_TOKEN_BEDROCK="tu_token"

model = LiteLLMModel(
    model_id="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
    aws_region_name="us-east-1",
)

# ==============================================================================
# FASE 1: PRE-AGENTE - Generación dinámica de personalidad
# ==============================================================================
# En lugar de tener la personalidad como texto estático, usamos el LLM para
# que la genere. Le damos el texto de El Principito y le pedimos que cree
# una personalidad que mezcle Pink Panther con El Principito.
#
# Ventajas de este enfoque:
# - La personalidad es más rica porque el LLM interpreta el libro completo
# - Cada ejecución puede producir matices ligeramente diferentes
# - Es más fácil cambiar la fuente (otro libro, otro personaje) sin reescribir
# - Demuestra el patrón de "agente que configura a otro agente"
# ==============================================================================

PRINCIPITO_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kbpink", "elprincipito.md"
)


def generate_personality() -> str:
    """
    Pre-agente: usa el LLM para generar la personalidad combinada de
    Pink Panther + El Principito, basándose en el texto del libro.

    Lee el archivo elprincipito.md y le pide al LLM que genere un prompt
    de personalidad que combine:
    - Lo que sabe públicamente de la Pink Panther (elegancia, jazz, humor sutil)
    - La filosofía y forma de ser del Principito (del texto proporcionado)

    Returns:
        String con la personalidad generada para usar como system prompt.
    """
    print("🧠 Pre-agente: Leyendo El Principito...")
    with open(PRINCIPITO_FILE, "r", encoding="utf-8") as f:
        principito_text = f.read()

    # Limitamos el texto si es muy largo para no exceder el contexto del LLM
    # 8000 caracteres cubren los capítulos más icónicos (la rosa, el zorro, lo esencial)
    # y reducen significativamente el consumo de tokens de entrada.
    if len(principito_text) > 8000:
        principito_text = principito_text[:8000] + "\n\n[... texto continúa ...]"

    print("🎭 Pre-agente: Generando personalidad combinada Pink Panther + Principito...")

    # Llamada directa al LLM usando litellm (sin agente, es solo generación de texto)
    # max_tokens limita la respuesta para optimizar costos y velocidad.
    # 500 tokens (~375 palabras en español) son suficientes para una personalidad concisa.
    response = litellm.completion(
        model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un experto en diseño de personajes y prompts para agentes de IA. "
                    "Tu trabajo es crear descripciones de personalidad detalladas y efectivas. "
                    "Sé CONCISO: genera personalidades en máximo 400 palabras."
                ),
            },
            {
                "role": "user",
                "content": f"""Necesito que generes un prompt de personalidad CONCISO para un agente de IA.
El agente debe ser una MEZCLA de dos personajes:

1. **La Pink Panther (Pantera Rosa)**: Usa tu conocimiento público del personaje.
   Es la pantera rosa de las películas y caricaturas: elegante, cool, silenciosa, 
   sofisticada, con humor sutil y físico, se mueve al ritmo del jazz (la famosa 
   música de Henry Mancini con saxofón), resuelve todo con ingenio y estilo, 
   nunca con fuerza. Es icónica, de color rosa, misteriosa y un poco traviesa.

2. **El Principito**: Basándote en el siguiente texto del libro de Antoine de 
   Saint-Exupéry, extrae su filosofía, forma de hablar, valores y personalidad:

--- INICIO DEL TEXTO DE EL PRINCIPITO ---
{principito_text}
--- FIN DEL TEXTO DE EL PRINCIPITO ---

Genera un prompt de personalidad en español que:
- Describa quién es este personaje híbrido (Pink Panther + Principito)
- Defina su esencia y valores (la mezcla de ambos)
- Explique cómo debe responder (tono, estilo, metáforas que usa)
- Incluya frases o ideas clave del Principito que debe usar
- Incluya elementos de la Pink Panther (elegancia, jazz, humor visual)
- Establezca reglas claras: responder en español, ser profundo pero accesible
- El formato debe ser en markdown con secciones claras

El prompt debe empezar con "## Tu Personalidad:" y ser CONCISO (máximo 400 palabras).
Debe ser suficientemente claro para que un LLM adopte esta personalidad de forma convincente.
Genera SOLO el prompt de personalidad, sin explicaciones adicionales.
IMPORTANTE: Sé breve y directo. Prioriza calidad sobre cantidad.""",
            },
        ],
        aws_region_name="us-east-1",
    )

    personality = response.choices[0].message.content
    print("✅ Pre-agente: Personalidad generada exitosamente.")
    print("-" * 70)
    print("📝 Personalidad generada (primeras líneas):")
    # Mostrar solo las primeras 5 líneas como preview
    for line in personality.split("\n")[:5]:
        if line.strip():
            print(f"   {line}")
    print("   ...")
    print("-" * 70)
    return personality


# Ejecutar el pre-agente para obtener la personalidad
print("=" * 70)
print("🎬 FASE 1: Pre-agente generando personalidad...")
print("=" * 70)

GENERATED_PERSONALITY = generate_personality()
# En smolagents 1.24.0, la personalidad se pasa con el parámetro "instructions"
# que se inyecta dentro del system prompt base del agente.
CUSTOM_INSTRUCTIONS = GENERATED_PERSONALITY

# ==============================================================================
# Funciones de carga y procesamiento de documentos
# ==============================================================================
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kbpink")


def load_markdown_documents(directory: str) -> list[dict]:
    """Carga todos los archivos .md del directorio."""
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
    print(f"📄 Cargados {len(documents)} documentos desde {directory}")
    return documents


def chunk_document(doc: dict, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Divide un documento en chunks con overlap."""
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


def build_vector_store(documents: list[dict]) -> tuple:
    """Construye la colección de ChromaDB con embeddings."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    print(f"🔪 Generados {len(all_chunks)} chunks de texto")

    print("🧠 Cargando modelo de embeddings...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["content"] for chunk in all_chunks]
    print("📐 Generando embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="pink_panther_kb",
        metadata={"description": "Knowledge base de historias de Pink Panther"},
    )
    collection.add(
        ids=[chunk["chunk_id"] for chunk in all_chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"filename": chunk["filename"]} for chunk in all_chunks],
    )
    print(f"✅ Base de datos vectorial creada con {collection.count()} documentos")
    return collection, embedding_model


# ==============================================================================
# Herramienta de retrieval para smolagents
# ==============================================================================
class RAGRetrieverTool(Tool):
    """Herramienta de búsqueda semántica en la KB de Pink Panther."""

    name = "knowledge_base_search"
    description = (
        "Busca información relevante en la base de conocimiento de historias de Pink Panther. "
        "Usa esta herramienta para encontrar fragmentos de texto relacionados con tu consulta. "
        "La base contiene historias sobre Pink Panther, Gus, filosofía, ciencia ficción, "
        "música, amor, y reflexiones sobre la humanidad y la tecnología."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "La consulta de búsqueda. Debe ser una frase descriptiva "
                "semánticamente cercana al contenido que buscas. "
                "Usa forma afirmativa en lugar de pregunta."
            ),
        }
    }
    output_type = "string"

    def __init__(self, collection, embedding_model, **kwargs):
        super().__init__(**kwargs)
        self.collection = collection
        self.embedding_model = embedding_model

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "La consulta debe ser un string"
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5,
        )
        if not results["documents"][0]:
            return "No se encontraron documentos relevantes."

        output = "\n📚 Documentos recuperados:\n"
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            output += f"\n{'='*60}\n"
            output += f"📄 Documento {i+1} (Fuente: {metadata['filename']})\n"
            output += f"{'='*60}\n"
            output += doc + "\n"
        return output


# ==============================================================================
# FASE 2: Inicialización del pipeline RAG
# ==============================================================================
print("\n" + "=" * 70)
print("🎬 FASE 2: Construyendo pipeline RAG...")
print("=" * 70)

documents = load_markdown_documents(DOCS_DIR)
collection, embedding_model = build_vector_store(documents)
retriever_tool = RAGRetrieverTool(collection, embedding_model)

# Crear el agente con la personalidad GENERADA por el pre-agente
agent = CodeAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=4,
    verbosity_level=2,
    instructions=CUSTOM_INSTRUCTIONS,  # <-- Personalidad generada dinámicamente
)

print("\n✅ Agente RAG con personalidad generada listo.")
print("🌟 'Lo esencial es invisible a los ojos...'\n")


# ==============================================================================
# Interfaz de chat con Gradio (con tabs)
# ==============================================================================
def chat(message: str, history: list) -> str:
    """
    Función que procesa cada mensaje del usuario.
    Recibe el mensaje y el historial de conversación,
    ejecuta el agente RAG y retorna la respuesta.
    """
    try:
        response = agent.run(message)
        return str(response)
    except Exception as e:
        return f"❌ Error al procesar la consulta: {str(e)}"


# Crear la interfaz con Tabs: uno para el chat y otro para la personalidad
with gr.Blocks(title="🌹✨ Pink Panther como El Principito - Chat RAG") as demo:
    gr.Markdown("# 🌹✨ Pink Panther como El Principito - Chat RAG")

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            chat_interface = gr.ChatInterface(
                fn=chat,
                description=(
                    "Chatea con Pink Panther que ha adoptado la sabiduría del Principito. "
                    "Pregunta sobre sus aventuras, reflexiones filosóficas, tecnología, música y más. "
                    "Recuerda: lo esencial es invisible a los ojos. 🌟"
                ),
                examples=[
                    "¿Qué piensa Pink Panther sobre la inteligencia artificial?",
                    "¿Cuál es la relación entre Pink Panther y Gus?",
                    "¿Qué historias de ciencia ficción se mencionan?",
                    "¿Qué reflexiones hay sobre el amor y la música?",
                    "¿Qué enseña Pink Panther sobre la amistad?",
                ],
            )

        with gr.Tab("🎭 Personalidad"):
            gr.Markdown(
                "![Pink Panther](https://upload.wikimedia.org/wikipedia/en/9/96/Pink_Panther.png)\n\n"
                "## Personalidad generada por el Pre-Agente\n\n"
                "Esta personalidad fue creada dinámicamente por el LLM al iniciar "
                "la aplicación, combinando su conocimiento de la Pink Panther con "
                "el texto de El Principito.\n\n---\n\n"
                f"{CUSTOM_INSTRUCTIONS}"
            )

if __name__ == "__main__":
    demo.launch()
