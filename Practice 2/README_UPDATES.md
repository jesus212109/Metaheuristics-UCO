# Resumen de Cambios: Algoritmo Genético Adaptativo (v3.0)

Este documento resume **todas las mejoras** realizadas en el proyecto para que el equipo esté alineado. Incluye los cambios de hoy y explica el impacto real de cada decisión.

---

## 🐛 Arreglos Críticos de Lógica (Bugs)

### 1. Selección por Torneo Corregida
Había un error de indexación: el torneo elegía 3 candidatos al azar, pero luego los comparaba por su **posición en la lista** en lugar de por su **fitness real**. Esto hacía que la selección fuera casi aleatoria. Ahora el torneo es 100% competitivo y justo, con un impacto directo en la calidad de la convergencia.

### 2. Evaluación (Cross-Validation) Fija con StratifiedKFold ⚠️ CAMBIO MÁS IMPORTANTE
Cambiamos el `cv=5` de sklearn por `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

**Por qué importa tanto:**
- El `cv=5` original tomaba los datos en orden secuencial. Como el dataset de vinos está parcialmente ordenado por calidad, cada fold era sesgado (unos contenían muchos vinos malos, otros muchos buenos).
- **Con shuffle=True**, cada fold tiene una mezcla aleatoria representativa.
- **Con StratifiedKFold**, cada fold tiene exactamente la misma proporción de clases buenas/malas.

**Impacto en los resultados:**

| Método | Antes (CV roto) | Después (CV correcto) |
|--------|-----------------|----------------------|
| Random Search  | ~0.740 | 0.767 |
| Grid Search    | ~0.739 | 0.812 |
| Algoritmo GA   | ~0.747 | **0.815** |

El GA ahora gana claramente al Grid Search con una evaluación honesta.

---

## 🧬 Mejoras en la Diversidad

### 3. Separación Inicial por Distancia de Hamming
La función `is_diverse()` garantiza que los 40 individuos de la Generación 0 sean distintos entre sí en al menos 2 parámetros. Evita que el algoritmo empiece "agrupado" en una sola zona del espacio de búsqueda.

### 4. Mutación Quirúrgica (Ruleta de Impacto)
Antes, la mutación destruía demasiados genes a la vez. Ahora usa una ruleta:
- 75% → cambia 1 gen (micro-ajuste fino)
- 20% → cambia 2 genes (salto intermedio)
- 5%  → cambia 3 genes (salto drástico, escape de mínimos locales)

---

## 🎯 Comportamiento Adaptativo (Pc y Pm)

### 5. Detección de Estancamiento y Cambio de Modo
El algoritmo monitoriza la media del fitness de la élite y ajusta las probabilidades:
- **Si mejora:** Sube `Pc` (Modo Explotación, refinar lo bueno).
- **Si se estanca 10 generaciones seguidas:** Sube `Pm` (Modo Exploración, buscar nuevo territorio).

### 6. Suelo y Techo de Seguridad (Opción B)
Para evitar que el algoritmo se destruya solo convirtiéndose en un Random Search:
- `Pc` nunca baja de **0.40** (el cruce siempre tiene peso)
- `Pm` nunca sube de **0.60** (la mutación no arrasa con todo)

### 7. Parámetros de Inicio Calibrados
- `Pc = 0.65`, `Pm = 0.35`: Sesgo inicial hacia la explotación, aprovechando la diversidad garantizada por Hamming desde el principio.
- `delta = 0.05`: Paso suave de ajuste para evitar oscilaciones bruscas.
- `stagnation_limit = 10`: Más paciencia antes de reaccionar (plateau real vs ruido).

---

## 🚀 Rendimiento y Escalabilidad

### 8. Caché de Fitness (Memoización)
Un diccionario global evita re-evaluar configuraciones idénticas. Combinado con el elitismo, acelera drásticamente las generaciones tardías cuando la población converge.

### 9. Elitismo (10%)
Los 3-4 mejores individuos de cada generación sobreviven intactos. El "techo de precisión" nunca retrocede.

### 10. Configuración Final
`pop_size=40`, `generations=50`, `elite_size=3`.

---

## 📊 Benchmark y Visualización

### 11. Script `benchmark.py`
Ejecuta los tres métodos **5 veces independientemente** y genera automáticamente 7 plots para la memoria:

| # | Archivo | Descripción |
|---|---------|-------------|
| 1 | `comparison_boxplot.png` | Violín + box: distribución de accuracy por método |
| 2 | `ga_convergence.png` | Curva de convergencia del GA (5 runs + media ± std) |
| 3 | `ga_adaptive_params.png` | Evolución de Pc/Pm por generación |
| 4 | `runs_bar_chart.png` | Accuracy por ejecución individual: RS vs GA |
| 5 | `rs_score_histogram.png` | Distribución de los 500 scores muestreados por RS |
| 6 | `gs_heatmap.png` | Mapa de calor del Grid Search (n_estimators × max_depth) |
| 7 | `eval_cost_comparison.png` | Coste computacional real por método |

Para lanzarlo esta noche (desde la raíz del proyecto):
```bash
python3 "Practice 2/src/benchmark.py"
```

---

## 📝 Documentación y Limpieza

- Todos los comentarios del código están al **100% en inglés**.
- Las funciones críticas están factorizadas: `is_diverse`, `init_population`, `generate_random_params`.
- Los reportes LaTeX (`report_es.tex` y `report_en.tex`) tienen portada, índice y una defensa académica profunda de cada decisión de diseño.

---

> **Resultado final con CV correcto:**
> - Random Search: `0.7667`
> - Grid Search: `0.8123`
> - **Algoritmo Genético: `0.8154` ← GANADOR**
