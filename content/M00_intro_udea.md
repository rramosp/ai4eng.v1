# Proyecto sustitutorio Modelos 1

<font color="red">Esta es información para los estudiantes matriculados en la materia de Modelos I, que ya han visto previamente la electiva de Inteligencia Artificial.</font>

## Estructura de proyecto

El objetivo del proyecto es completar la formación anterior llevando un modelo predictivo a un estado listo para que sea integrado en sistema de producción. 

El proyecto tendrá tres fases:

**FASE 1. Modelo predictivo**:
- Escoge un challenge de Kaggle. Mejor si es una competición y no un dataset, ya que las competiciones son más completas y tienen más contribuciones de código. <font color="red">Se recomiendo escoger algún dataset relacionado con el transporte personal (Taxis, Uber, etc.)</font>, para que haya una posibilidad futura de ser integrado en los proyectos de la Escuela de Software. Pero no es obligatorio.
- Desarrolla o replica un modelo predictivo para el challenge. Puedes desarrollarlo tú mismo, o puedes seleccionar algún modelo que alguien ya haya realizado mirando la parte de **code** de Kaggle. 
- No te preocupes si las predicciones no son muy precisas. Lo importante es que emita predicciones.

**FASE 2. Despligue en container**:
- Configura un contenedor de Docker con todas las librerías necesarias para correr el modelo.
- El contenedor ha de tener dos scripts:
  - `predict.py`: que dado un conjunto de datos de entrada como un fichero `csv`, emita una predicción para cada dato de entrada, usando un modelo previamente almacenado en disco.
  - `train.py`: que dado un conjunto de entrenamiento (datos más etiquetas), entrene de nuevo el modelo y guarde una versión nueva del mismo.

**FASE 3. API REST**:
- Crea una aplicación REST en un script **python** `apirest.py` (p.ej. con `flask`) que exponga dos `endpoints`:
  - `predict`: que con un dato nuevo devuelve su predicción
  - `train`: que lanza un proceso de entrenamiento, con unso datos de entrenamiento estándar.

## Entregas

Tendrás que entregar tu proyecto en un repositorio de github, que contenga:
- Un directorio `fase-1`, con al menos n notebook que muestre cómo se entrena y se predice con el modelo
- Un directorio `fase-2`, con los scripts `predict.py` y `train.py` y un `Dockerfile` para crear el contenedor con las librerías y los scripts anteriores incluidos  
- Un directorio `fase-3`, con los scripts anteriores, más `apirest.py`, más un `Dockerfile` nuevo que extienda el anterior para instalar todo lo necesario para el API REST.

Añade un `README.md` al repositorio github donde se describan los pasos para ejecutar cada elemento de cada fase.

## Fechas

    FASE 1: 26 de septiembre
    FASE 2: 24 de octubre
    FASE 3: 14 de noviembre

## Evaluación

    FASE 1: 40%
    FASE 2: 40%
    FASE 3: 20%


Para cada fase se evaluará:
  - 10% que los elementos de la entrega estén presentes (ficherso, github, etc.)
  - 50% que siguiendo el `README.md` se ejecuten correctamente los elementos de las entregan. Se seguirán paso a paso las instrucciones. Se penalizará cualquier paso que haya que dar que no esté descrito en el fichero. 
  - 40% que tanto el notebook, como los scripts como los dockerfiles estén bien documentados. Es decir:
      - Que el notebook explique brevemente qué hace cada celda.
      - Que los scripts y las funciones tengan sus correspondientes `doctrings`
      - Que los dockerfiles tengan un comentario adjunto a cada línea de código.


