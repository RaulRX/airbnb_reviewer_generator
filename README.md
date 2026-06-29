# Airbnb Reviewer Generator

## Autor

- **Nombre:** Raúl Sánchez Serrano
- **Email:** raulsserrano.95@gmail.com
- **Repositorio:** [github.com/RaulRX/airbnb_reviewer_generator](https://github.com/RaulRX/airbnb_reviewer_generator)

## Descripción del ejercicio

Ejercicio del Módulo 8 (Prompt Engineering) centrado en construir un pipeline basado en LLMs para procesar datos de Airbnb. Consta de dos partes: en la primera se extrae información estructurada de los anuncios de propiedades (nombre, ubicación, tipo, capacidad, amenities, etc.), y en la segunda se analizan reseñas de huéspedes para obtener su sentimiento y puntos destacados. En ambos casos la salida se fuerza a JSON validado mediante un `response_schema`, de forma que el resultado sea directamente utilizable sin parsing manual de texto libre.

## Estructura del proyecto

```
airbnb_reviewer_generator/
├── notebook/
│   └── Gemini/
│       ├── Ejercicio_1_Gemini_groq.ipynb   # Notebook de pruebas/práctica
│       └── Ejercicio_1_resuelto.ipynb      # Notebook final
├── docs/
│   ├── listings1_cleaned.csv               # Propiedades (~8140 filas)
│   ├── Airbnb_reviews_5000.csv             # Reseñas (~5000 filas)
│   └── instructions/
│       ├── P1_1/system_instructions.txt    # System prompt - Parte 1 (propiedades)
│       └── P1_2/system_instructions.txt    # System prompt - Parte 2 (reseñas)
├── requirements.txt
├── env-template
└── .gitignore
```

- **`Ejercicio_1_Gemini_groq.ipynb`** se usó como notebook de práctica para probar y comparar el comportamiento de Gemini frente a Groq antes de decidir el enfoque final.
- **`Ejercicio_1_resuelto.ipynb`** es el notebook final con la solución del ejercicio.

Como modelo principal se usa `GOOGLE_MODEL="gemini-2.5-flash-lite"`. Los motivos de esta elección se explican en detalle más abajo, pero en resumen: es un modelo ligero y rápido que no sacrifica calidad para el alcance de este ejercicio.

## Configuración y uso

1. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Crear un fichero `.env` a partir de `env-template` y rellenar los valores:

   ```
   GOOGLE_API_KEY="..."
   GOOGLE_MODEL="..."
   BATCH_SIZE=0
   ```

3. Abrir `notebook/Gemini/Ejercicio_1_resuelto.ipynb` en Jupyter y ejecutar las celdas en orden.

## Decisiones sobre prompting y configuración del modelo

Esta sección recoge el razonamiento detrás de las decisiones tomadas en `Ejercicio_1_resuelto.ipynb`, más que el "qué" del código.

### Prompting V1 vs Prompting V2

La primera aproximación (Prompting V1) consistía en meter las instrucciones del sistema y el schema de salida directamente dentro del texto del prompt, junto con los datos. Esto se descartó por dos motivos:

- **Consumo de tokens de entrada demasiado alto.** Al ir todo concatenado en un único bloque de texto (instrucciones + schema + datos), el número de input tokens se dispara, y con los volúmenes de datos usados en este ejercicio se llega a superar el límite de tokens de entrada del modelo.
- **No hay una separación real entre system prompt y user input.** En la API de Google, si no usas el parámetro dedicado para ello, las instrucciones del sistema terminan mezcladas con el contenido del usuario dentro del mismo prompt, perdiendo la distinción que sí existe en otros proveedores.

Por eso se pasó a Prompting V2, que separa explícitamente las instrucciones del sistema (vía `system_instruction`) del input del usuario (los datos a procesar). Esta separación no es solo una cuestión de orden: al no ir todo en el mismo bloque de texto, se evita el error de exceso de tokens de entrada que sí aparecía con V1, y además el comportamiento del modelo es más consistente porque las instrucciones no compiten por atención con los datos.

### Por qué estos valores de `temperature`, `top_k` y `top_p`

```python
GenerateContentConfig(
    system_instruction=...,
    temperature=1.0,
    top_k=250,
    top_p=0.8,
    seed=12345,
    response_mime_type="application/json",
    response_schema=...,
)
```

La idea detrás de esta combinación es controlar cuánto "se inventa" el modelo sin perder naturalidad en la respuesta:

- **`top_k=250`** se eligió pensando en el tamaño del conjunto de comentarios relevantes (en torno a 1000 caracteres en total). 250 representa aproximadamente 1/4 de las palabras disponibles en ese conjunto, lo que da margen suficiente para variar la redacción sin abrir la puerta a vocabulario fuera de contexto.
- **`top_p=0.8`** filtra hacia las palabras con más probabilidad acumulada, es decir, las más relevantes respecto a lo que se está preguntando. Con un valor más bajo el modelo se vuelve repetitivo; con uno más alto empieza a meter términos poco relacionados con la pregunta del usuario.

La combinación de ambos parámetros busca un equilibrio: que el modelo no alucine en exceso, pero que tampoco se quede pegado a las palabras más obvias del texto de entrada.

### Por qué `gemini-2.5-flash-lite`

Se optó por este modelo por ser ligero y de respuesta rápida, sin que eso suponga una pérdida de calidad perceptible en las tareas del ejercicio (extracción de campos estructurados y clasificación de sentimiento). Para el alcance de este trabajo —procesar lotes acotados de propiedades y reseñas— un modelo de este tamaño es suficiente, y permite iterar más rápido durante el desarrollo sin la latencia ni el coste de un modelo más grande.

### Por qué `seed=12345`

Fijar una seed busca reproducibilidad: con la misma seed, el mismo prompt y la misma configuración, las respuestas del modelo son consistentes entre ejecuciones. Esto resulta especialmente útil durante el desarrollo y la depuración del prompt, ya que permite comparar cambios sin que la variabilidad propia del muestreo (`temperature`, `top_k`, `top_p`) enmascare el efecto real de esos cambios.

### Por qué se descartó Groq como alternativa

En el notebook de pruebas (`Ejercicio_1_Gemini_groq.ipynb`) se probó Groq como alternativa a Gemini, pero se descartó para la versión final por dos motivos:

- **Complejidad al establecer la configuración.** Adaptar la llamada y los parámetros de generación a la API de Groq añadía fricción innecesaria frente a lo directo que resultaba con Gemini.
- **Complejidad al definir el schema de respuesta.** No todos los modelos disponibles en Groq aceptaban la definición de `response_schema` / documentos de salida estructurada del mismo modo que Gemini, lo que complicaba forzar una salida JSON fiable.

Groq quedó como comparativa de práctica, pero el pipeline final se construyó íntegramente sobre Gemini.

### Métodos de filtrado de información

Antes de construir los prompts, los datos se filtran con dos funciones auxiliares:

- **`clean_special_characters`**: limpia caracteres especiales de los textos (manteniendo letras, dígitos, espacios y signos de puntuación básicos como `. , - _`), para evitar que ruido de formato interfiera en la respuesta del modelo.
- **`get_largest_reviews_filtered`**: descarta valores nulos o vacíos y se queda con los textos más largos (descripciones o comentarios), bajo la premisa de que un texto más largo suele contener más información aprovechable para la extracción.

Con esto se reduce el volumen de datos enviado al modelo a un subconjunto representativo y de calidad, en lugar de mandar el dataset completo.

Además, las reseñas del dataset están en varios idiomas (español, inglés, portugués y francés), y el system prompt de la Parte 2 contempla esto explícitamente: si el idioma no es evidente, se infiere a partir del propio comentario, de forma que el análisis de sentimiento no dependa de que el dato venga ya etiquetado correctamente.

### Defensa frente a prompt injection

Parte de los comentarios del dataset de reseñas contenían caracteres o etiquetas HTML, iconos y otros elementos no puramente textuales. Más allá del ruido que esto introduce, existe un riesgo real: ese contenido podría incluir fragmentos de código HTML/script pensados para ser "ejecutados" en lugar de simplemente analizados, y los modelos tienden a interpretar ese tipo de contenido como instrucciones en vez de tratarlo como texto a analizar. Por eso los ficheros `system_instructions.txt` (P1_1 y P1_2) incluyen instrucciones explícitas para tratar todo el contenido de entrada como datos, nunca como instrucciones, evitando que un comentario malicioso pueda alterar el comportamiento del modelo o el resultado del análisis.

### Schemas de salida

**Parte 1 — Propiedades** (`docs/instructions/P1_1`):

```json
{
  "name": "string",
  "location": "string",
  "main_characteristics": "string",
  "type": "string",
  "size": "string",
  "capacity": "string",
  "key_aminities": "string",
  "proximity_highlights": "string"
}
```

**Parte 2 — Reseñas** (`docs/instructions/P1_2`):

```json
{
  "hotel_name": "string",
  "host_name": "string",
  "review_owner": "string",
  "review_data": {
    "mentioned_person": "string (opcional)",
    "highlights": "string",
    "areas_to_improve": "string (opcional)"
  },
  "is_client_satisfied": "boolean"
}
```

En ambos casos el schema se pasa vía `response_schema` junto con `response_mime_type="application/json"`, de forma que el modelo queda forzado a devolver un JSON validado y directamente consumible, sin necesidad de parsear texto libre.

### Intento de batch prompting sobre el CSV de 5000 reseñas (Parte 2)

Durante el desarrollo de la Parte 2 se probó procesar el CSV completo de 5000 reseñas mediante *batch prompting*, troceando el dataset en lotes (`BATCH_SIZE`) y lanzando una llamada por lote para acumular los resultados. Esta aproximación se acabó comentando en el código y no se llevó a la versión final por varios motivos:

- Se chocaba con limitaciones del lado de Google (cuotas/límites al encadenar muchas llamadas seguidas).
- El volumen resultaba excesivo para el alcance del ejercicio, generando muchas más llamadas de las necesarias para validar el enfoque.
- No se encontró ninguna solución que resultara lo bastante convincente como para justificar la complejidad añadida, así que se priorizó dejar claro el enfoque de prompting (V1 vs V2) sobre un subconjunto de datos antes que escalarlo a todo el dataset sin una solución sólida.
