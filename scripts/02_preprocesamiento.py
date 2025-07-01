import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import requests
from io import StringIO
import joblib
import os

print("🔧 PREPROCESAMIENTO DE DATOS CARDIOVASCULARES")
print("=" * 50)
print("Universidad Privada Antenor Orrego")
print("=" * 50)

# Cargar datos
print("\n📊 Cargando dataset...")
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/cardio_train-tr3GWz87HWlFsgFKUqnDaAfEW3pyYY.csv"

try:
    response = requests.get(url)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text), sep=';')
    
    print(f"✅ Dataset cargado: {data.shape[0]:,} registros")
    
    # Crear directorio para guardar modelos y escaladores
    os.makedirs('models', exist_ok=True)
    
    print("\n🧹 LIMPIEZA DE DATOS")
    print("-" * 30)
    
    # 1. Convertir edad de días a años
    data['age'] = data['age'] / 365.25
    print("✅ Edad convertida de días a años")
    
    # 2. Crear IMC
    data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
    print("✅ IMC calculado")
    
    # 3. Limpiar valores anómalos en presión arterial
    print(f"Registros antes de limpieza: {len(data):,}")
    
    # Filtrar presión arterial anómala
    data = data[
        (data['ap_hi'] >= 80) & (data['ap_hi'] <= 200) &
        (data['ap_lo'] >= 40) & (data['ap_lo'] <= 120) &
        (data['ap_hi'] > data['ap_lo'])  # Sistólica > Diastólica
    ]
    
    # Filtrar IMC anómalo
    data = data[(data['bmi'] >= 15) & (data['bmi'] <= 50)]
    
    # Filtrar altura y peso anómalos
    data = data[
        (data['height'] >= 140) & (data['height'] <= 220) &
        (data['weight'] >= 40) & (data['weight'] <= 200)
    ]
    
    print(f"Registros después de limpieza: {len(data):,}")
    print(f"Registros eliminados: {len(data) - data.shape[0]:,}")
    
    # 4. Crear variables derivadas
    print("\n🔬 INGENIERÍA DE CARACTERÍSTICAS")
    print("-" * 30)
    
    # Presión arterial media
    data['map'] = (data['ap_hi'] + 2 * data['ap_lo']) / 3
    
    # Presión de pulso
    data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']
    
    # Categorías de presión arterial (según AHA)
    def categorize_bp(systolic, diastolic):
        if systolic < 120 and diastolic < 80:
            return 0  # Normal
        elif systolic < 130 and diastolic < 80:
            return 1  # Elevada
        elif (systolic >= 130 and systolic < 140) or (diastolic >= 80 and diastolic < 90):
            return 2  # Hipertensión Etapa 1
        else:
            return 3  # Hipertensión Etapa 2
    
    data['bp_category'] = data.apply(lambda x: categorize_bp(x['ap_hi'], x['ap_lo']), axis=1)
    
    # Categorías de IMC
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 0  # Bajo peso
        elif bmi < 25:
            return 1  # Normal
        elif bmi < 30:
            return 2  # Sobrepeso
        else:
            return 3  # Obesidad
    
    data['bmi_category'] = data['bmi'].apply(categorize_bmi)
    
    # Puntuación de riesgo de estilo de vida
    data['lifestyle_risk'] = data['smoke'] + data['alco'] + (1 - data['active'])
    
    # Interacciones importantes
    data['age_bmi'] = data['age'] * data['bmi']
    data['age_bp'] = data['age'] * data['ap_hi']
    
    print("✅ Variables derivadas creadas:")
    print("   - Presión arterial media (MAP)")
    print("   - Presión de pulso")
    print("   - Categorías de presión arterial")
    print("   - Categorías de IMC")
    print("   - Puntuación de riesgo de estilo de vida")
    print("   - Interacciones edad-IMC y edad-presión")
    
    # 5. Seleccionar características finales
    print("\n📋 SELECCIÓN DE CARACTERÍSTICAS")
    print("-" * 30)
    
    # Características para el modelo
    feature_columns = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active',
        'bmi', 'map', 'pulse_pressure', 'bp_category', 'bmi_category',
        'lifestyle_risk', 'age_bmi', 'age_bp'
    ]
    
    X = data[feature_columns].copy()
    y = data['cardio'].copy()
    
    print(f"Características seleccionadas: {len(feature_columns)}")
    print(f"Variable objetivo: cardio (0: Sin enfermedad, 1: Con enfermedad)")
    
    # 6. División de datos
    print("\n📊 DIVISIÓN DE DATOS")
    print("-" * 30)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]:,} registros ({X_train.shape[0]/len(data)*100:.1f}%)")
    print(f"Conjunto de validación: {X_val.shape[0]:,} registros ({X_val.shape[0]/len(data)*100:.1f}%)")
    print(f"Conjunto de prueba: {X_test.shape[0]:,} registros ({X_test.shape[0]/len(data)*100:.1f}%)")
    
    # 7. Normalización
    print("\n⚖️ NORMALIZACIÓN DE DATOS")
    print("-" * 30)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Datos normalizados con StandardScaler")
    
    # Guardar el escalador
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Escalador guardado en models/scaler.pkl")
    
    # 8. Guardar datos procesados
    print("\n💾 GUARDANDO DATOS PROCESADOS")
    print("-" * 30)
    
    # Guardar conjuntos de datos
    np.save('models/X_train.npy', X_train_scaled)
    np.save('models/X_val.npy', X_val_scaled)
    np.save('models/X_test.npy', X_test_scaled)
    np.save('models/y_train.npy', y_train.values)
    np.save('models/y_val.npy', y_val.values)
    np.save('models/y_test.npy', y_test.values)
    
    # Guardar nombres de características
    with open('models/feature_names.txt', 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    
    # Guardar estadísticas del dataset
    stats = {
        'total_records': len(data),
        'train_records': len(X_train),
        'val_records': len(X_val),
        'test_records': len(X_test),
        'features': len(feature_columns),
        'positive_class_ratio': y.mean(),
        'feature_names': feature_columns
    }
    
    import json
    with open('models/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("✅ Datos guardados exitosamente:")
    print("   - Conjuntos de entrenamiento, validación y prueba")
    print("   - Escalador entrenado")
    print("   - Nombres de características")
    print("   - Estadísticas del dataset")
    
    # 9. Resumen final
    print("\n📈 RESUMEN DEL PREPROCESAMIENTO")
    print("-" * 30)
    print(f"Dataset original: {data.shape[0]:,} registros")
    print(f"Características finales: {len(feature_columns)}")
    print(f"Distribución de clases:")
    print(f"  - Sin enfermedad: {(1-y.mean())*100:.1f}%")
    print(f"  - Con enfermedad: {y.mean()*100:.1f}%")
    print(f"Datos listos para entrenamiento de modelos 🚀")
    
except Exception as e:
    print(f"❌ Error en el preprocesamiento: {str(e)}")
