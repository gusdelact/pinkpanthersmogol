# 🚀 Deploy a Hugging Face Spaces

## Paso 1: Crear el Space

1. Ve a https://huggingface.co/new-space
2. Configura:
   - **Space name**: `pink-panther-principito-rag`
   - **SDK**: Gradio
   - **Hardware**: CPU basic (gratuito)
3. Click en **Create Space**

---

## Paso 2: Clonar el repo de GitHub y subirlo al Space

```bash
# Clonar el repo fuente
git clone https://github.com/gusdelact/pinkpanthersmogol.git
cd pinkpanthersmogol

# Cambiar el remote a tu Space de Hugging Face
git remote set-url origin https://huggingface.co/spaces/TU_USUARIO/pink-panther-principito-rag

# Subir
git push origin main
```

Si te pide credenciales, usa tu username de Hugging Face y un token con permisos de escritura.

---

## Paso 3: Configurar los secretos

Ve a tu Space → **Settings** → **Repository secrets** y agrega:

| Name | Value |
|------|-------|
| `AWS_BEARER_TOKEN_BEDROCK` | Tu token de Amazon Bedrock |
| `HF_TOKEN` | Tu token de Hugging Face |

---

## Paso 4: Verificar

1. Ve a tu Space: `https://huggingface.co/spaces/TU_USUARIO/pink-panther-principito-rag`
2. Espera a que se construya (2-5 minutos)
3. Si hay errores, revisa la pestaña **Logs**

---

## Notas

- NUNCA subas tokens en el código. Usa siempre Repository Secrets.
- Los Spaces gratuitos se duermen tras inactividad. El primer acceso después tarda en cargar.
- Si necesitas reconstruir, ve a Settings → Factory reboot.
