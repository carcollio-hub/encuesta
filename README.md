# Dashboard SCD — Encuesta de socios

## Cómo ejecutarlo localmente

1. Instala Python 3.9+ si no lo tienes.
2. Abre una terminal en esta carpeta y ejecuta:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. Se abrirá automáticamente en tu navegador (normalmente `http://localhost:8501`).

## Uso

- El panel viene precargado con `Encuesta.xlsx` (los mismos 159 datos del informe).
- Puedes subir un archivo nuevo desde el panel lateral ("Subir archivo de respuestas") para actualizar los datos sin tocar el código — debe tener las mismas columnas/preguntas.
- Filtra por género desde el panel lateral.
- Pestañas: Perfil, Procesos de inscripción, Comunicación, Recomendación (NPS), Texto libre (nubes de palabras).

## Publicarlo online (opcional, gratis)

La forma más simple es Streamlit Community Cloud:
1. Sube esta carpeta a un repositorio de GitHub.
2. Entra a share.streamlit.io, conecta el repo y selecciona `app.py`.
3. En minutos tendrás un link público para compartir con tu equipo en SCD.
