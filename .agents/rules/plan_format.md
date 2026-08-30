# Formato de Planes

Cuando el usuario pida "armar un plan" o "crear un plan" (implementation_plan.md o similar en el directorio docs/ u otro lado), DEBES escribir el plan SIEMPRE como una secuencia de pasos a seguir numerados (ej. 1, 2, 3...). 

El documento debe contener:
- Una lista secuencial y accionable de los pasos a realizar.
- Cada paso debe incluir los comandos exactos (bash, git, etc.) o los detalles de implementación necesarios para ese momento del flujo (preparación del entorno, instalación de dependencias con `pip install -e ".[dev]"`, creación de ramas, commits, ejecución de tests y linters, merges, etc.).
- Siempre incluir, cuando haga falta o al inicio de la preparación del entorno, la sentencia de instalación/actualización de dependencias (`pip install -e ".[dev]"`).
- No utilices un formato de resumen general sin orden; todo debe estar estructurado paso a paso.
