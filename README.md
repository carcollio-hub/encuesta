# Dashboard SCD — Encuesta de socios

## Cómo ejecutarlo localmente

1. Instala Python 3.9+ si no lo tienes.
2. Abre una terminal en esta carpeta y ejecuta:

   ```bash
   pip install -r requirements.txt
   python -m streamlit run app.py
   ```

3. Se abrirá automáticamente en tu navegador (normalmente `http://localhost:8501`).

## Uso

- El panel viene precargado con `Encuesta.xlsx` (los 159 datos de la encuesta) — no requiere subir ningún archivo.
- Filtra por género desde el panel lateral.
- Pestañas: Sentimiento, Puntos de fricción, Oportunidades de mejora, Perfil, NPS, Datos, Conclusiones.
- Si en el futuro quieres actualizar los datos, reemplaza el archivo `Encuesta.xlsx` de esta misma carpeta por uno nuevo con las mismas columnas/preguntas, y vuelve a ejecutar la app.

## Publicarlo online (opcional, gratis)

La forma más simple es Streamlit Community Cloud:
1. Sube esta carpeta a un repositorio de GitHub.
2. Entra a share.streamlit.io, conecta el repo y selecciona `app.py`.
3. En minutos tendrás un link público para compartir con tu equipo en SCD.
