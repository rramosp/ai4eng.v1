# Información 2021.2 - UdeA

<font color="red">Esta es información para los matriculados en el curso ofrecido por el Departamento de Ingeniería de Sistemas, Facultad de Ingeniería,
Universidad de Antioquia, Medellín.</font>

## 2021 Semestre 2 - Universidad de Antioquia

**Plataforma de autocorrectores para los laboratorios**

- [REGISTRATE AQUÍ EN LA PLATAFORMA](https://m5knaekxo6.execute-api.us-west-2.amazonaws.com/dev-v0001/rlxmooc/web/request_invitation/ai4eng.v1.udea/20212) para poder someter las soluciones a los talleres del curso.
- [ACCESO A LA PLATAFORMA](https://m5knaekxo6.execute-api.us-west-2.amazonaws.com/dev-v0001/rlxmooc/web/login) para ver tus calificaciones y descargar el certificado cuando completes el curso.

Revisa estos dos vídeos para ver cómo se interacciona con los materiales del curso:

- **Trabajando con los materiales del curso** [Video 13mins](https://youtu.be/Rg0_9EBtUIc)
- **Talleres y plataforma de autocorrección** [Video 13mins](https://youtu.be/eISlF6k0y58)



**Grupos de proyecto**

- [REGISTRO DE GRUPOS DE PROYECTO](https://forms.gle/vYdPzTBvnDcepFtS9) (solo un registro por grupo)


## Horario de clases
    
        MARTES 10:00-12:00          JUEVES 10:00-12:00

<br/>

Las sesiones se realizarán por Zoom a través del siguiente enlace 

<center><big><a href="https://udea.zoom.us/j/97304122517">https://udea.zoom.us/j/97304122517</a></big></center>

<br/>

**Grabaciones**: Accede en [este enlace](http://ingeniaudea.edu.co/zoom-recordings/recordings/docenciaingenia54@udea.edu.co/97304122517/2021-02-28) al repositorio de grabaciones de las sesiones sincrónicas del curso.


## Discusiones, Q&A, Incidencias

Indícanos cualquier duda, sugerencia o incidencia, o inicia una discusión con la comunidad del curso en el siguiente foro. Las discusiones son en abierto. Si quieres participar tendrás que utilizar tu usuario Github.

<center><big><a href="https://github.com/rramosp/ai4eng.v1/discussions">FORO DE DISCUSIONES DEL CURSO</a></big></center>


Plantea tu cuestión o inquietud en la categoría del módulo pertinente. Aspectos generales, administrativos, fechas, evaluaciones, etc. los puedes formular en la categoría **UDEA ai4eng 2021.2**.

Durante las sesiones le daremos prioridad <font color='red'><b>A LAS PREGUNTAS que se FORMULEN PREVIAMENTE por EL FORO</b></font> en el orden en el que aparezcan.


## Grupo whatsapp estudiantes


https://chat.whatsapp.com/IOSb9lorLutDrrUqoZvbia


<font color="red"><b>NO SE RESPONDERÁN PREGUNTAS o INQUETUDES SOBRE EL CURSO EN ESTE GRUPO de WHATSAPP. SOLO POR EL FORO de DISCUSIONES</b></font>


## Evaluación

    10% (0.5 puntos) PROYECTO PRIMERA ENTREGA
    50% (2.5 puntos) LABS
    10% (0.5 puntos) AI4Everyone
    30% (1.5 puntos) PROYECTO ENTREGA FINAL


## Fechas de entregas

        05/dic/2021    REGISTRO GRUPOS DE PROYECTO
        31/dic/2021    LABS MODULOS 1-2
        31/dic/2021    PROYECTO ENTREGA 1
        13/feb/2022    LABS MODULOS 3-4S
        20/mar/2022    AI4Everyone
        30/mar/2022    LABS MODULOS 5-6-7
         9/abr/2022    PROYECTO ENTREGA 2
<br/>

**Calendario oficial**

       16/nov/2021  Inicio de clases
        2/abr/2022  Fin clases
      4-9/abr/2022  Exámenes finales
       17/abr/2022  Fecha límite cierre notas finales
    18-23/abr/2022  Habilitación y validación
       25/abr/2022  Terminación oficial


## Proyecto

Tendrás que hacer un proyecto de analítica de datos para el cual deberás:

1. Definir un problema predictivo
2. Obtener un dataset para resolverlo
3. Realizar el preprocesado y limpieza de datos
4. Encontrar los mejores hiperparámetros para DOS algoritmos predictivos
5. Encontrar los mejores hiperparámetros para DOS combinaciones de algoritmo no supervisado + algoritmo predictivo
6. Realizar las curvas de aprendizaje  para cada uno de los cuatro casos anteriores
7. Realizar una evaluación diagnóstica que contenga:
    1. Para cada uno de los cuatro casos anteriores un diagnóstico de overfitting/bias etc.
    1. Una recomendación justificada de qué pasos a seguir si tuvieras que intentar mejorar el desempeño obtenido.
    1. Una evaluación sobre los retos y condiciones para desplegar en producción un modelo (como establecerías el nivel de desempeño mínimo para desplegar en producción, cómo se desplegaría en producción, cómo serían los procesos de monitoreo del desempeño en producción)

**Podrá hacerse individual o formarse grupos de 2 o 3 estudiantes**

#### Recomendaciones

Te recomendamos que:

- Mires en [www.kaggle.com](https://www.kaggle.com) por ideas si no sabes qué hacer.
- Verifiques que los datos están disponibles antes de escoger tu proyecto.
- Estimes los requerimientos computacionales para generar los modelos que necesites. Reduce el alcance de tu proyecto si lo necesitas (menos datos, menos clases, etc.).
- Realices una primera iteración cuanto antes. Es decir, que llegues a tener un primer modelo **sencillo** produciendo predicciones. Implementa en esta primera iteración estrictamente lo que necesites para tener un modelo. El objetivo es resolver la mayoría de los problemas técnicos que te puedan surgir para ya, después, enfocarte en todo lo que quieras hacer en las siguientes iteraciones (preprocesado de datos, otros modelos, etc.)



#### Entregas del proyecto

Tendrás que hacer dos entregas del proyecto:

- **ENTREGA 1**: Un archivo llamado **PROYECTO_FASE1.pdf** en el que (1) describas el problema predictivo a resolver, (2) el dataset que vas a utilizar, (3) las métricas de desempeño requeridas (de machine learning y de negocio); y (4) un primer criterio sobre cual sería el desempeño deseable en producción. 

Dos ejemplos del punto (4) anterior:

- **Ejemplo 1**: nuestro modelo de predicción de la patología X en pacientes debería de tener un porcentaje de acierto >80%, pero también un false negative rate <5%, ya que es una patología grave y es preferible no fallar una detección de un paciente que verdaderamente tiene la patología, aunque eso implique que aumente el número de falsos positivos.

- **Ejemplo 2**: según el departamento de marketing de cierta empresa, un modelo de predicción del siguiente producto que compre un cliente debería de tener un porcentaje de acierto de al menos 50%, ya que se usará el modelo para sugerir recomendaciones a los usuarios. Si el porcentaje de acierto es menor sería contraproducente porque perderíamos ventas.

Como en cualquier proyecto de analítica, esto supone un **primer** criterio, que probablemente se refine o modifique según se avanza en el proyecto, se entiende mejor el posible desempeño de los modelos, con el cliente se va definiendo cómo se usan los modelos en producción/operación, etc.


- **ENTREGA 2**: Un archivo llamado **PROYECTO_FASE2.pdf** en el que describas el resto de la ejecucióndel proyecto, con la evidencia necesaria. Tendrás que incluir una carpeta llamada **PROYECTO_FASE2_MATERIALES** donde incluyas todas las herramientas necesarias para reproducir los resultados que obtuviste (muestras de los datasets, notebooks, instrucciones de uso de las herramientas que hayas usado, etc.)

**TODAS LAS ENTREGAS HAN DE DEPOSITARSE EN LA CARPETA DRIVE DE CADA ESTUDIANTE**.




## AI for Everyone

Deberás de completar el curso [AI for Everyone](https://www.deeplearning.ai/ai-for-everyone/) y hacer un resumen del mismo. El resumen deberá contener:

- tres páginas como máximo
- un párrafo por cada módulo del curso que decriba los puntos más importantes del mismo
- un pantallazo en el que se vea que has visualizado los videos
- una apreciación por tu parte acerca de la aplicabilidad de lo visto en el curso en proyectos relacionados con tu área de conocimiento.

Deberás de depositar el resumen en un fichero llamado **RESUMEN_AI4EVERYONE.pdf** en la carpeta compartida del curso.

## EL NOMBRADO DE ARCHIVOS ES ESTRICTO

<font color="red"><b>CUALQUIER ARCHIVO CON UN NOMBRE DISTINTO SERÁ IGNORADO, AUNQUE SEA ENTREGADO ANTES DE LAS FECHAS LÍMITE</b></font>
