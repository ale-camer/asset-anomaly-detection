# Formato de Planes

Cuando el usuario pida "armar un plan" o "crear un plan" (por ejemplo, en el directorio `docs/`), DEBES escribir el plan utilizando exactamente la misma estructura que `docs/issue_1_plan.md`. 

El documento debe seguir este formato exacto:

1. **Título:** `# Issue #<Número>: <Título>`
2. **Introducción:** Un breve párrafo describiendo el objetivo del issue.
3. **Sección Principal:** `## Pasos a Seguir`
4. **Pasos numerados con subtítulos:** Cada paso debe ser un subtítulo H3 (ej. `### 1. Preparar el Entorno y la Rama de Trabajo`).
5. **Bloques de código para comandos:** Debajo de la descripción de cada paso, los comandos bash o git deben ir dentro de un bloque de código ````bash ... ````, en lugar de estar sueltos en el texto.
6. **Contenido Obligatorio en los pasos:**
   - **Paso 1:** Crear la rama desde `develop`. (La instalación de dependencias ya NO es obligatoria aquí).
   - **Actualización de Dependencias:** Si el issue requiere agregar librerías a `pyproject.toml`, se debe hacer un paso posterior (ej. Paso 2) para modificarlas y luego sí incluir el comando `pip install -e ".[dev]"`. Si no se modifican dependencias, NO incluir este comando.
   - **Pasos intermedios:** Detalles de implementación precisos.
   - **Verificación:** Ejecución de tests (`python -m pytest -o addopts="" <archivos_test>`), linter (`ruff check . --fix`) y tipado (`mypy src/ tests/`).
   - **Commit:** Comandos `git add` y `git commit`.
   - **Integración:** Hacer merge a `develop` (o crear la PR) y `git push`.
