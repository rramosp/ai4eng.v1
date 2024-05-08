# Proyecto sustitutorio Modelos 1

<font color="red">Esta es información para los estudiantes matriculados en la materia de Modelos I, que ya han visto previamente la electiva de Inteligencia Artificial.</font>

## Estructura de proyecto

El objetivo del proyecto es completar la formación anterior llevando un modelo predictivo a un estado listo para que sea integrado en sistema de producción. 


El proyecto tendrá tres fases:

**FASE 1. Modelo predictivo**:
- Escoge un challenge de Kaggle. Mejor si es una competición y no un dataset, ya que las competiciones son más completas y tienen más contribuciones de código. <font color="red">Se recomiendo escoger algún dataset relacionado con el transporte personal (Taxis, Uber, etc.)</font>, para que haya una posibilidad futura de ser integrado en los proyectos de la Escuela de Software. Pero no es obligatorio.
- Desarrolla o replica un modelo predictivo para el challenge. Puedes desarrollarlo tú mismo, o puedes seleccionar algún modelo que alguien ya haya realizado mirando la parte de **code** de Kaggle. 
- No te preocupes si las predicciones no son muy precisas. Lo importante es que emita predicciones.

**FASE 2. Despliegue en container**:
- Configura un contenedor de Docker con todas las librerías necesarias para correr el modelo.
- El contenedor ha de tener dos scripts:
  - `predict.py`: que dado un conjunto de datos de entrada como un fichero `csv`, emita una predicción para cada dato de entrada, usando un modelo previamente almacenado en disco.
  - `train.py`: que dado un conjunto de entrenamiento (datos más etiquetas), entrene de nuevo el modelo y guarde una versión nueva del mismo.
- Para estos dos scripts puedes guiarte por el ejemplo en [https://github.com/rramosp/sklearn_scripts](https://github.com/rramosp/sklearn_scripts)

**FASE 3. API REST**:
- Crea una aplicación REST en un script **python** `apirest.py` (p.ej. con `flask`) que exponga dos `endpoints`:
  - `predict`: que con un dato nuevo devuelve su predicción
  - `train`: que lanza un proceso de entrenamiento, con unso datos de entrenamiento estándar.
- Para esta fase puedes guiarte con este repo [https://github.com/rramosp/restapiexample](https://github.com/rramosp/restapiexample)
    

<br/>

## Sesiones temáticas

- Experiencias en ejecución de proyectos de IA <a href='https://www.youtube.com/watch?v=Wpj80tZXZwc'>Video 1h 17mins</a> (29 Ago 2023)
- Introducción a docker <a href='https://udea.zoom.us/rec/play/vOwql6zvWfFWmYrlEPqAA9noBGTS_LqGCVgiacnwzDMDbWR0OSRSp4C2plW0JfLsGHSDvNbyEPgve7u1.vYoWWj4Lg46AADpt?canPlayFromShare=true&from=my_recording&continueMode=true&componentName=rec-play&originRequestUrl=https%3A%2F%2Fudea.zoom.us%2Frec%2Fshare%2F3uWxP0umRyJRXv-6NexPVtVXjW97-CJxszC6ZQStJ4IbNnYMPes4XWPoZ0pOrWxg.F-GyXrDZgL9aQSnr'>Grabación de la sesión sincrónica</a> (24 oct 2023)

<br/>

## Entregas

El proyecto se podrá realizar **individualmente** o **por parejas**. En cualquier caso, cada persona es responsable de su entrega. Si es en parejas, **ambas personas** han de hacer su entrega por separado, aunque el contenido sea el mismo. Si un miembro de una pareja no hace la entrega, no tendrá nota, independientemente de si su pareja entregó o no.

Tendrás que entregar tu proyecto en un repositorio de github, que contenga:
- Un directorio `fase-1`, con al menos n notebook que muestre cómo se entrena y se predice con el modelo
- Un directorio `fase-2`, con los scripts `predict.py` y `train.py` y un `Dockerfile` para crear el contenedor con las librerías y los scripts anteriores incluidos  
- Un directorio `fase-3`, con los scripts anteriores, más `apirest.py`, más un `Dockerfile` nuevo que extienda el anterior para instalar todo lo necesario para el API REST.

Añade un `README.md` al repositorio github donde se describan los pasos para ejecutar cada elemento de cada fase.

## Formularios para las entregas

- Fase 1: [Formulario](https://forms.gle/aEQEgwAdzh2uiKFk6)
- Fase 2: [Formulario](https://forms.gle/c3dVeBWkL8a1rdKy9)
- Fase 3: disponible próximamente

## Fechas

    FASE 1: 24 de marzo
    FASE 2: 28 de abril
    FASE 3: 26 de mayo

## Evaluación

    FASE 1: 40%
    FASE 2: 40%
    FASE 3: 20%


Para cada fase se evaluará:
  - 10% que los elementos de la entrega estén presentes (ficheros, github, etc.)
  - 50% que siguiendo el `README.md` se ejecuten correctamente los elementos de las entregas. Se seguirán paso a paso las instrucciones. Se penalizará cualquier paso que haya que dar que no esté descrito en el fichero. 
  - 40% que tanto el notebook, como los scripts como los dockerfiles estén bien documentados. Es decir:
      - Que el notebook explique brevemente qué hace cada celda.
      - Que los scripts y las funciones tengan sus correspondientes `docstrings`
      - Que los dockerfiles tengan un comentario adjunto a cada línea de código.


