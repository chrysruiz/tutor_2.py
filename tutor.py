import gradio as gr
import google.generativeai as genai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
import ast
import difflib

genai.configure(api_key="AIzaSyAiZI7ZJj5cN-gHa2XuVadDeajNS1R5TJE")
#genai.configure(api_key="AIzaSyARIrZVEBarwic4Z4dXrPKXxcDQkORvklc")# api cuenta secundaria
 #genai.configure(api_key="AIzaSyCy-UAY3_sm8QiQjcO9mfaFaIbg5aOvL2k") # Reemplaza con tu API KEY apy de cuenta principal

# Modelo del dominio
lecciones = {
    "variables": {"dificultad": "básico", "conceptos_relacionados": [], "contenido": "Explicación sobre variables..."},
    "clases": {"dificultad": "intermedio", "conceptos_relacionados": ["variables"], "contenido": "Explicación sobre clases..."},
    "interfaces": {"dificultad": "avanzado", "conceptos_relacionados": ["clases"], "contenido": "Explicación sobre interfaces..."},
    # Agrega más lecciones
}

# Modelo del estudiante
perfil_estudiante = {
    "conceptos": {},
    "lecciones_completadas": [],
    "estilo_aprendizaje": "visual",  # O "auditivo", "kinestésico"
    "preferencias": {
        "nivel_dificultad": "intermedio",
        "tipo_ejercicios": "codificación"  # O "opción múltiple"
    },
    "progreso_general": {
        "puntuacion_total": 0,
        "lecciones_vistas": 0
    },
    "errores": {},  # {"concepto": {"tipo_error": cantidad}}
    "estilos_aprendizaje_detectados": {"visual": 0, "auditivo": 0, "kinestésico": 0}
}

# Historial
historial = pd.DataFrame(columns=["pregunta", "respuesta", "dificultad", "calificacion", "conceptos_aprendidos"])

# Modelo de clasificación (como en el ejemplo anterior)
data = pd.DataFrame({
    "pregunta": ["¿Qué es una variable?", "¿Cómo se implementa una interfaz?"],
    "dificultad": ["básico", "avanzado"]
})
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["pregunta"])
y = data["dificultad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

def reentrenar_modelo():
    """Reentrena el modelo de clasificación con los datos del historial."""
    global model, vectorizer
    if len(historial) > 5:  # Reentrena solo si hay suficientes datos
        X_historial = vectorizer.fit_transform(historial["pregunta"])
        y_historial = historial["dificultad"]
        model.fit(X_historial, y_historial)

def registrar_error(concepto, tipo_error):
    """Registra un error cometido por el estudiante."""
    global perfil_estudiante
    if concepto not in perfil_estudiante["errores"]:
        perfil_estudiante["errores"][concepto] = {}
    if tipo_error not in perfil_estudiante["errores"][concepto]:
        perfil_estudiante["errores"][concepto][tipo_error] = 0
    perfil_estudiante["errores"][concepto][tipo_error] += 1

def detectar_estilo_aprendizaje(respuesta):
    """Detecta el estilo de aprendizaje del estudiante (ejemplo simplificado)."""
    global perfil_estudiante
    if "imagen" in respuesta.lower() or "diagrama" in respuesta.lower():
        perfil_estudiante["estilos_aprendizaje_detectados"]["visual"] += 1
    elif "audio" in respuesta.lower() or "explicación" in respuesta.lower():
        perfil_estudiante["estilos_aprendizaje_detectados"]["auditivo"] += 1
    elif "ejercicio práctico" in respuesta.lower() or "codificar" in respuesta.lower():
        perfil_estudiante["estilos_aprendizaje_detectados"]["kinestésico"] += 1
    # Determinar el estilo predominante (ejemplo simplificado)
    estilo_predominante = max(perfil_estudiante["estilos_aprendizaje_detectados"], key=perfil_estudiante["estilos_aprendizaje_detectados"].get)
    perfil_estudiante["estilo_aprendizaje"] = estilo_predominante

def generar_respuesta(pregunta, calificacion=None):
    global historial, perfil_estudiante
    modelo_gemini = genai.GenerativeModel('gemini-1.5-pro')
    
    # Añadir el prompt con las instrucciones
    prompt_con_instrucciones = f"Eres un tutor especializado en lenguajes de programación. Solo responde a preguntas relacionadas con este tema. Pregunta: {pregunta}"

    response = modelo_gemini.generate_content(prompt_con_instrucciones)
    dificultad = model.predict(vectorizer.transform([pregunta]))[0]
    # Extraer conceptos (lógica simple)
    conceptos_aprendidos = []
    for concepto in lecciones:
        if concepto.lower() in pregunta.lower() or concepto.lower() in response.text.lower():
            conceptos_aprendidos.append(concepto)
    # Actualizar perfil del estudiante
    for concepto in conceptos_aprendidos:
        if concepto not in perfil_estudiante["conceptos"]:
            perfil_estudiante["conceptos"][concepto] = {"dominio": 0, "intentos": 0, "ultima_interaccion": datetime.datetime.now()}
        perfil_estudiante["conceptos"][concepto]["intentos"] += 1
        perfil_estudiante["conceptos"][concepto]["ultima_interaccion"] = datetime.datetime.now()
        # Actualizar dominio basado en la calificación (ejemplo simplificado)
        if calificacion == "Bien":
            perfil_estudiante["conceptos"][concepto]["dominio"] = min(1, perfil_estudiante["conceptos"][concepto]["dominio"] + 0.2)
        elif calificacion == "Mal":
            perfil_estudiante["conceptos"][concepto]["dominio"] = max(0, perfil_estudiante["conceptos"][concepto]["dominio"] - 0.1)
    # Actualizar historial
    nueva_fila = pd.DataFrame({
        "pregunta": [pregunta],
        "respuesta": [response.text],
        "dificultad": [dificultad],
        "calificacion": [calificacion],
        "conceptos_aprendidos": [conceptos_aprendidos]
    })
    historial = pd.concat([historial, nueva_fila], ignore_index=True)
    if len(historial) % 5 == 0:
        reentrenar_modelo()
    # Lógica del motor de tutoría mejorada
    lecciones_recomendadas = []
    conceptos_para_recomendar = conceptos_aprendidos if conceptos_aprendidos else [list(lecciones.keys())[0]]
    for concepto_pregunta in conceptos_para_recomendar:
        for concepto_dominio, datos in lecciones.items():
            if concepto_dominio in [concepto_pregunta] + lecciones[concepto_pregunta]["conceptos_relacionados"]:
                if concepto_dominio not in perfil_estudiante["conceptos"]:
                    lecciones_recomendadas.append(concepto_dominio)
    # Planificación de lecciones dinámica (ejemplo simplificado)
    if calificacion == "Mal" and conceptos_aprendidos:
        concepto_fallido = conceptos_aprendidos[0]
        if concepto_fallido in lecciones:
            conceptos_previos = lecciones[concepto_fallido]["conceptos_relacionados"]
            lecciones_recomendadas.extend(conceptos_previos)
    # Estrategias de enseñanza adaptativas (ejemplo simplificado)
    estrategia_enseñanza = "explicación"  # Predeterminado
    if perfil_estudiante["estilo_aprendizaje"] == "visual":
        estrategia_enseñanza = "diagrama"
    elif perfil_estudiante["estilo_aprendizaje"] == "kinestésico":
        estrategia_enseñanza = "ejercicio práctico"
    # Retroalimentación
    retroalimentacion = ""
    if calificacion == "Mal":
       retroalimentacion = "Te recomiendo revisar las siguientes lecciones: " + ", ".join(lecciones_recomendadas)
    return response.text + f"\n\n(Dificultad: {dificultad})\n\nLecciones recomendadas: {lecciones_recomendadas}\n\n{retroalimentacion}"

def calificar_respuesta(calificacion, pregunta):
    """Califica la respuesta y actualiza el perfil del estudiante."""
    global perfil_estudiante
    historial.loc[historial['pregunta'] == pregunta, 'calificacion'] = calificacion
    # Corrección: usar apply() para convertir valores nulos en listas vacías
    conceptos_aprendidos = historial.loc[historial['pregunta'] == pregunta, 'conceptos_aprendidos'].apply(lambda x: x if isinstance(x, list) else []).iloc[0]
    if conceptos_aprendidos:
        for concepto in conceptos_aprendidos:
            if concepto not in perfil_estudiante["conceptos"]:
                perfil_estudiante["conceptos"][concepto] = {"dominio": 0, "intentos": 0, "ultima_interaccion": datetime.datetime.now()}
            perfil_estudiante["conceptos"][concepto]["intentos"] += 1
            perfil_estudiante["conceptos"][concepto]["ultima_interaccion"] = datetime.datetime.now()
            if calificacion == "Bien":
                perfil_estudiante["conceptos"][concepto]["dominio"] = min(1, perfil_estudiante["conceptos"][concepto]["dominio"] + 0.2)
            elif calificacion == "Mal":
                perfil_estudiante["conceptos"][concepto]["dominio"] = max(0, perfil_estudiante["conceptos"][concepto]["dominio"] - 0.1)
    return "Calificación registrada. Gracias."
 
def verificar_respuesta_usuario(respuesta_usuario, pregunta_original):
    """Verifica la respuesta del usuario con retroalimentación detallada y ejemplos."""
    global perfil_estudiante
    modelo_gemini = genai.GenerativeModel('gemini-1.5-pro')

    respuesta_modelo = modelo_gemini.generate_content(pregunta_original).text
    prompt_solucion = f"Genera una solución sencilla para el siguiente problema, adaptando el lenguaje a la respuesta del usuario si es posible: {pregunta_original}"
    solucion_referencia = modelo_gemini.generate_content(prompt_solucion).text

    dificultad_pregunta = model.predict(vectorizer.transform([pregunta_original]))[0]

    try:
        # Intentar analizar la sintaxis (esto podría fallar para lenguajes no compatibles con ast)
        # Por lo tanto, lo dejamos como una verificación opcional
        # ast.parse(respuesta_usuario)
        similitud = difflib.SequenceMatcher(None, respuesta_usuario, solucion_referencia).ratio()

        if similitud > 0.8:
            retroalimentacion = "¡Correcto! Tu respuesta es correcta.\n\n"
            calificacion = "Bien"
            prompt_explicacion = f"Explica por qué la siguiente respuesta es correcta para una pregunta de nivel {dificultad_pregunta}: {respuesta_usuario}. Proporciona un ejemplo claro y sencillo, adaptando el lenguaje al código del usuario."
            explicacion = modelo_gemini.generate_content(prompt_explicacion).text
            retroalimentacion += explicacion
        else:
            retroalimentacion = "Tu respuesta es sintácticamente correcta, pero no coincide con la solución esperada.\n\n"
            calificacion = "Mal"
            prompt_explicacion = f"Explica la solución correcta para la siguiente pregunta de nivel {dificultad_pregunta}: {pregunta_original}. Proporciona un ejemplo claro y sencillo, adaptando el lenguaje al código del usuario. Solución esperada:\n{solucion_referencia}"
            explicacion = modelo_gemini.generate_content(prompt_explicacion).text
            retroalimentacion += explicacion
    except SyntaxError:
        retroalimentacion = "Tu respuesta tiene errores de sintaxis.\n\n"
        calificacion = "Mal"
        prompt_explicacion = f"Explica los errores de sintaxis en el siguiente código y cómo corregirlos, adaptando la explicación al nivel {dificultad_pregunta} y al lenguaje del código del usuario: {respuesta_usuario}"
        explicacion = modelo_gemini.generate_content(prompt_explicacion).text
        retroalimentacion += explicacion
    except Exception as e:
        retroalimentacion = f"Ocurrió un error al analizar tu código: {e}\n\n"
        calificacion = "Mal"
        prompt_explicacion = f"Explica posibles errores en el siguiente código, adaptando la explicación al nivel {dificultad_pregunta} y al lenguaje del código del usuario: {respuesta_usuario}"
        explicacion = modelo_gemini.generate_content(prompt_explicacion).text
        retroalimentacion += explicacion

    historial.loc[historial['pregunta'] == pregunta_original, 'calificacion'] = calificacion
    conceptos_aprendidos = historial.loc[historial['pregunta'] == pregunta_original, 'conceptos_aprendidos'].apply(lambda x: x if isinstance(x, list) else []).iloc[0]

    if conceptos_aprendidos:
        for concepto in conceptos_aprendidos:
            if concepto not in perfil_estudiante["conceptos"]:
                perfil_estudiante["conceptos"][concepto] = {"dominio": 0, "intentos": 0, "ultima_interaccion": datetime.datetime.now()}
            perfil_estudiante["conceptos"][concepto]["intentos"] += 1
            perfil_estudiante["conceptos"][concepto]["ultima_interaccion"] = datetime.datetime.now()
            if calificacion == "Bien":
                perfil_estudiante["conceptos"][concepto]["dominio"] = min(1, perfil_estudiante["conceptos"][concepto]["dominio"] + 0.2)
            elif calificacion == "Mal":
                perfil_estudiante["conceptos"][concepto]["dominio"] = max(0, perfil_estudiante["conceptos"][concepto]["dominio"] - 0.1)

    return retroalimentacion, gr.JSON(value=perfil_estudiante)

with gr.Blocks(title="Tutor Inteligente Empresarial", theme=gr.themes.Soft()) as iface:
    gr.Markdown("# Chatbot de Aprendizaje Empresarial")
    with gr.Tab("Chat"):
        pregunta_input = gr.Textbox(lines=2, placeholder="Escribe tu pregunta aquí...", label="Pregunta")
        respuesta_output = gr.Textbox(label="Respuesta del Tutor", interactive=False)
        with gr.Row():
            enviar_btn = gr.Button("Enviar", variant="primary")
            calificacion_radio = gr.Radio(["Bien", "Mal"], label="¿Fue útil la respuesta?")
            calificar_btn = gr.Button("Calificar")
        with gr.Accordion("Retroalimentación del Usuario"):
            respuesta_usuario_input = gr.Textbox(lines=2, placeholder="Escribe tu respuesta aquí...", label="Tu Respuesta")
            retroalimentacion_usuario_output = gr.Textbox(label="Retroalimentación del Tutor", interactive=False)
            verificar_respuesta_btn = gr.Button("Verificar Respuesta", variant="secondary")

        enviar_btn.click(fn=generar_respuesta, inputs=pregunta_input, outputs=respuesta_output)
        calificar_btn.click(fn=calificar_respuesta, inputs=[calificacion_radio, pregunta_input], outputs=gr.Textbox())
        verificar_respuesta_btn.click(fn=verificar_respuesta_usuario, inputs=[respuesta_usuario_input, pregunta_input], outputs=[retroalimentacion_usuario_output, gr.JSON()])

    with gr.Tab("Perfil del Estudiante"):
        perfil_output = gr.JSON(value=perfil_estudiante)

iface.launch()
