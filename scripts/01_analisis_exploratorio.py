import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Configuración para gráficos en español
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
sns.set_palette("husl")

print("🫀 ANÁLISIS EXPLORATORIO DE DATOS - RIESGO CARDIOVASCULAR")
print("=" * 60)
print("Universidad Privada Antenor Orrego")
print("=" * 60)

# Cargar datos desde la URL
print("\n📊 Cargando dataset cardiovascular...")
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/cardio_train-tr3GWz87HWlFsgFKUqnDaAfEW3pyYY.csv"

try:
    response = requests.get(url)
    response.raise_for_status()
    
    # El dataset usa ';' como separador según el schema
    data = pd.read_csv(StringIO(response.text), sep=';')
    
    print(f"✅ Dataset cargado exitosamente!")
    print(f"📏 Dimensiones: {data.shape[0]:,} filas × {data.shape[1]} columnas")
    
    # Información básica del dataset
    print("\n📋 INFORMACIÓN GENERAL DEL DATASET")
    print("-" * 40)
    print(data.info())
    
    print("\n📊 PRIMERAS 5 FILAS")
    print("-" * 40)
    print(data.head())
    
    print("\n📈 ESTADÍSTICAS DESCRIPTIVAS")
    print("-" * 40)
    print(data.describe())
    
    # Análisis de valores faltantes
    print("\n🔍 ANÁLISIS DE VALORES FALTANTES")
    print("-" * 40)
    missing_values = data.isnull().sum()
    missing_percent = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({
        'Columna': missing_values.index,
        'Valores Faltantes': missing_values.values,
        'Porcentaje (%)': missing_percent.values
    })
    print(missing_df)
    
    # Convertir edad de días a años
    data['age_years'] = data['age'] / 365.25
    
    # Análisis de la variable objetivo
    print("\n🎯 ANÁLISIS DE LA VARIABLE OBJETIVO (CARDIO)")
    print("-" * 40)
    cardio_counts = data['cardio'].value_counts()
    cardio_percent = data['cardio'].value_counts(normalize=True) * 100
    
    print("Distribución de casos:")
    print(f"Sin enfermedad cardiovascular (0): {cardio_counts[0]:,} ({cardio_percent[0]:.1f}%)")
    print(f"Con enfermedad cardiovascular (1): {cardio_counts[1]:,} ({cardio_percent[1]:.1f}%)")
    
    # Análisis por género
    print("\n👥 ANÁLISIS POR GÉNERO")
    print("-" * 40)
    gender_analysis = pd.crosstab(data['gender'], data['cardio'], normalize='index') * 100
    print("Porcentaje de enfermedad cardiovascular por género:")
    print(f"Género 1: {gender_analysis.loc[1, 1]:.1f}%")
    print(f"Género 2: {gender_analysis.loc[2, 1]:.1f}%")
    
    # Análisis por rangos de edad
    print("\n🎂 ANÁLISIS POR RANGOS DE EDAD")
    print("-" * 40)
    data['age_group'] = pd.cut(data['age_years'], 
                              bins=[0, 40, 50, 60, 70, 100], 
                              labels=['<40', '40-50', '50-60', '60-70', '70+'])
    
    age_analysis = pd.crosstab(data['age_group'], data['cardio'], normalize='index') * 100
    print("Porcentaje de enfermedad cardiovascular por grupo de edad:")
    for age_group in age_analysis.index:
        print(f"{age_group} años: {age_analysis.loc[age_group, 1]:.1f}%")
    
    # Análisis de presión arterial
    print("\n🩺 ANÁLISIS DE PRESIÓN ARTERIAL")
    print("-" * 40)
    
    # Detectar valores anómalos en presión arterial
    ap_hi_outliers = data[(data['ap_hi'] < 80) | (data['ap_hi'] > 200)].shape[0]
    ap_lo_outliers = data[(data['ap_lo'] < 40) | (data['ap_lo'] > 120)].shape[0]
    
    print(f"Valores anómalos en presión sistólica: {ap_hi_outliers:,}")
    print(f"Valores anómalos en presión diastólica: {ap_lo_outliers:,}")
    
    # Análisis de IMC
    print("\n⚖️ ANÁLISIS DE ÍNDICE DE MASA CORPORAL (IMC)")
    print("-" * 40)
    data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
    
    def classify_bmi(bmi):
        if bmi < 18.5:
            return 'Bajo peso'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Sobrepeso'
        else:
            return 'Obesidad'
    
    data['bmi_category'] = data['bmi'].apply(classify_bmi)
    bmi_analysis = pd.crosstab(data['bmi_category'], data['cardio'], normalize='index') * 100
    
    print("Porcentaje de enfermedad cardiovascular por categoría de IMC:")
    for category in bmi_analysis.index:
        print(f"{category}: {bmi_analysis.loc[category, 1]:.1f}%")
    
    # Análisis de factores de estilo de vida
    print("\n🚬 ANÁLISIS DE FACTORES DE ESTILO DE VIDA")
    print("-" * 40)
    
    lifestyle_factors = ['smoke', 'alco', 'active']
    lifestyle_names = ['Fumador', 'Consume Alcohol', 'Físicamente Activo']
    
    for factor, name in zip(lifestyle_factors, lifestyle_names):
        factor_analysis = pd.crosstab(data[factor], data['cardio'], normalize='index') * 100
        print(f"{name}:")
        print(f"  No (0): {factor_analysis.loc[0, 1]:.1f}% con enfermedad cardiovascular")
        print(f"  Sí (1): {factor_analysis.loc[1, 1]:.1f}% con enfermedad cardiovascular")
    
    # Correlaciones
    print("\n🔗 MATRIZ DE CORRELACIONES")
    print("-" * 40)
    
    # Seleccionar variables numéricas para correlación
    numeric_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 
                   'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'bmi']
    
    correlation_matrix = data[numeric_cols].corr()
    
    # Correlaciones más fuertes con la variable objetivo
    cardio_correlations = correlation_matrix['cardio'].abs().sort_values(ascending=False)
    print("Variables más correlacionadas con enfermedad cardiovascular:")
    for var, corr in cardio_correlations.items():
        if var != 'cardio':
            print(f"{var}: {corr:.3f}")
    
    print("\n✅ ANÁLISIS EXPLORATORIO COMPLETADO")
    print("📊 Datos listos para preprocesamiento y modelado")
    
except Exception as e:
    print(f"❌ Error al cargar los datos: {str(e)}")
