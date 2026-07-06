# Plan de Exploración — Hotel Bookings

Dataset: `hotel_bookings_cleaned.csv` | 115,596 registros | 34 columnas

---

## 0. Librerías
- pandas, numpy, matplotlib, seaborn, plotly

---

## 1. Carga de datos
- Leer el CSV con pandas
- Mostrar primeras filas (`head`)

---

## 2. Inspección inicial
- Forma del dataset (`shape`)
- Tipos de datos (`dtypes`)
- Valores nulos (`isnull().sum()`)
- Estadísticas descriptivas (`describe()`)
- Valores únicos por columna categórica

---

## 3. Limpieza y validación
- Confirmar que no hay duplicados
- Revisar columnas con valores atípicos (ej. `adr` negativo o cero)
- Validar consistencia: `total_nights = stays_in_weekend_nights + stays_in_week_nights`
- Convertir `reservation_status_date` a datetime
- Confirmar que `is_family` e `is_canceled` son binarios

---

## 4. Análisis univariado
### Numéricas
- Distribución de `lead_time`, `adr`, `total_nights`
- Histogramas + boxplots para detectar outliers

### Categóricas
- Frecuencia de `hotel`, `meal`, `market_segment`, `distribution_channel`, `deposit_type`, `customer_type`, `country` (top 10)

---

## 5. Análisis bivariado
- Tasa de cancelación por tipo de hotel (`hotel` vs `is_canceled`)
- Tasa de cancelación por `market_segment`, `deposit_type`, `customer_type`
- `adr` promedio por tipo de hotel y por mes
- `lead_time` vs `is_canceled` (¿más anticipación = más cancelaciones?)
- `total_nights` vs `adr`

---

## 6. Análisis temporal
- Reservas por año y mes (`arrival_date_year`, `arrival_date_month`)
- Estacionalidad del `adr`
- Cancelaciones a lo largo del tiempo

---

## 7. Análisis geográfico
- Top 10 países de origen de los huéspedes
- Tasa de cancelación por país

---

## 8. Segmentación de clientes
- Huéspedes repetidos vs nuevos (`is_repeated_guest`)
- Familias vs no familias (`is_family`)
- Tipos de cliente (`customer_type`) y su comportamiento

---

## 9. Correlaciones
- Heatmap de correlación entre variables numéricas
- Variables más correlacionadas con `is_canceled` y con `adr`

---

## 10. Insights y conclusiones
- Resumen de los hallazgos más relevantes
- Patrones de cancelación
- Segmentos más rentables
- Recomendaciones basadas en los datos
