import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🔍 ANÁLISIS DE EXPLICABILIDAD CON SHAP")
print("=" * 45)
print("Universidad Privada Antenor Orrego")
print("IA Explicable para Predicción Cardiovascular")
print("=" * 45)

try:
    # Cargar datos y modelo
    print("\n📊 Cargando datos y modelo...")
    
    X_test = np.load('models/X_test.npy')
    y_test = np.load('models/y_test.npy')
    
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Cargar el mejor modelo (asumiendo que es XGBoost)
    best_model = joblib.load('models/best_model.pkl')
    
    print(f"✅ Datos cargados: {X_test.shape[0]:,} registros de prueba")
    print(f"✅ Modelo cargado exitosamente")
    print(f"✅ Características: {len(feature_names)}")
    
    # Convertir a DataFrame para mejor manejo
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    print("\n🧠 INICIALIZANDO EXPLICADOR SHAP")
    print("-" * 40)
    
    # Crear explicador SHAP
    # Para modelos tree-based como XGBoost y Random Forest
    if hasattr(best_model, 'predict_proba'):
        explainer = shap.Explainer(best_model, X_test_df[:100])  # Usar muestra para eficiencia
        print("✅ Explicador SHAP inicializado (TreeExplainer)")
    else:
        # Para otros modelos
        explainer = shap.KernelExplainer(best_model.predict_proba, X_test_df[:100])
        print("✅ Explicador SHAP inicializado (KernelExplainer)")
    
    # Calcular valores SHAP para una muestra
    print("\n📈 CALCULANDO VALORES SHAP")
    print("-" * 40)
    
    # Usar una muestra más pequeña para eficiencia
    sample_size = min(500, len(X_test_df))
    X_sample = X_test_df.sample(n=sample_size, random_state=42)
    
    print(f"Calculando SHAP para {sample_size} muestras...")
    shap_values = explainer(X_sample)
    
    print("✅ Valores SHAP calculados exitosamente")
    
    print("\n🎯 ANÁLISIS DE IMPORTANCIA GLOBAL")
    print("-" * 40)
    
    # Importancia global de características
    feature_importance = np.abs(shap_values.values).mean(0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("Top 10 características más importantes (SHAP):")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<15}: {row['importance']:.4f}")
    
    print("\n📊 ANÁLISIS DE CASOS ESPECÍFICOS")
    print("-" * 40)
    
    # Analizar casos específicos
    # Caso de alto riesgo
    high_risk_idx = np.argmax(best_model.predict_proba(X_sample)[:, 1])
    high_risk_case = X_sample.iloc[high_risk_idx]
    high_risk_prob = best_model.predict_proba(X_sample.iloc[[high_risk_idx]])[:, 1][0]
    
    print(f"🔴 CASO DE ALTO RIESGO (Probabilidad: {high_risk_prob:.1%})")
    print("Características principales:")
    
    # Obtener valores SHAP para este caso
    case_shap = shap_values[high_risk_idx]
    case_contributions = pd.DataFrame({
        'feature': feature_names,
        'value': high_risk_case.values,
        'shap_value': case_shap.values,
        'contribution': np.abs(case_shap.values)
    }).sort_values('contribution', ascending=False)
    
    for i, (_, row) in enumerate(case_contributions.head(5).iterrows()):
        direction = "↑" if row['shap_value'] > 0 else "↓"
        print(f"   {i+1}. {row['feature']}: {row['value']:.2f} {direction} ({row['shap_value']:+.3f})")
    
    # Caso de bajo riesgo
    low_risk_idx = np.argmin(best_model.predict_proba(X_sample)[:, 1])
    low_risk_case = X_sample.iloc[low_risk_idx]
    low_risk_prob = best_model.predict_proba(X_sample.iloc[[low_risk_idx]])[:, 1][0]
    
    print(f"\n🟢 CASO DE BAJO RIESGO (Probabilidad: {low_risk_prob:.1%})")
    print("Características principales:")
    
    case_shap_low = shap_values[low_risk_idx]
    case_contributions_low = pd.DataFrame({
        'feature': feature_names,
        'value': low_risk_case.values,
        'shap_value': case_shap_low.values,
        'contribution': np.abs(case_shap_low.values)
    }).sort_values('contribution', ascending=False)
    
    for i, (_, row) in enumerate(case_contributions_low.head(5).iterrows()):
        direction = "↑" if row['shap_value'] > 0 else "↓"
        print(f"   {i+1}. {row['feature']}: {row['value']:.2f} {direction} ({row['shap_value']:+.3f})")
    
    print("\n💡 INTERPRETACIÓN DE VALORES SHAP")
    print("-" * 40)
    print("• Valores SHAP positivos (+): Aumentan el riesgo cardiovascular")
    print("• Valores SHAP negativos (-): Disminuyen el riesgo cardiovascular")
    print("• Magnitud del valor: Indica la fuerza del impacto")
    print("• Suma de valores SHAP = Predicción - Valor base del modelo")
    
    print("\n🔍 PATRONES IDENTIFICADOS")
    print("-" * 40)
    
    # Analizar patrones en los valores SHAP
    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    
    # Características que más aumentan el riesgo
    positive_impact = shap_df.mean().sort_values(ascending=False)
    print("Características que MÁS AUMENTAN el riesgo (promedio):")
    for i, (feature, impact) in enumerate(positive_impact.head(5).items()):
        if impact > 0:
            print(f"   {i+1}. {feature}: +{impact:.4f}")
    
    # Características que más disminuyen el riesgo
    negative_impact = shap_df.mean().sort_values(ascending=True)
    print("\nCaracterísticas que MÁS DISMINUYEN el riesgo (promedio):")
    for i, (feature, impact) in enumerate(negative_impact.head(5).items()):
        if impact < 0:
            print(f"   {i+1}. {feature}: {impact:.4f}")
    
    print("\n📋 RECOMENDACIONES BASADAS EN SHAP")
    print("-" * 40)
    
    # Generar recomendaciones basadas en el análisis SHAP
    recommendations = []
    
    # Analizar las características más importantes
    top_features = importance_df.head(5)['feature'].tolist()
    
    if 'age' in top_features:
        recommendations.append("La edad es un factor de riesgo no modificable, pero otros factores pueden compensar su efecto")
    
    if 'ap_hi' in top_features or 'ap_lo' in top_features:
        recommendations.append("La presión arterial es un factor crítico - su control puede reducir significativamente el riesgo")
    
    if 'bmi' in top_features:
        recommendations.append("El índice de masa corporal tiene gran impacto - mantener peso saludable es fundamental")
    
    if 'smoke' in top_features:
        recommendations.append("El tabaquismo es un factor de alto impacto modificable - dejar de fumar reduce dramáticamente el riesgo")
    
    if 'active' in top_features:
        recommendations.append("La actividad física regular es protectora contra enfermedades cardiovasculares")
    
    print("Recomendaciones clínicas basadas en el análisis:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print("\n✅ ANÁLISIS DE EXPLICABILIDAD COMPLETADO")
    print("🎯 El modelo es interpretable y sus decisiones pueden explicarse")
    print("📊 Los valores SHAP proporcionan insights valiosos para la práctica clínica")
    print("💡 La IA explicable aumenta la confianza en las predicciones médicas")
    
    # Guardar resultados del análisis SHAP
    shap_analysis = {
        'global_importance': importance_df.to_dict('records'),
        'high_risk_case': {
            'probability': float(high_risk_prob),
            'top_contributors': case_contributions.head(5).to_dict('records')
        },
        'low_risk_case': {
            'probability': float(low_risk_prob),
            'top_contributors': case_contributions_low.head(5).to_dict('records')
        },
        'recommendations': recommendations,
        'interpretation_guide': {
            'positive_shap': "Aumenta el riesgo cardiovascular",
            'negative_shap': "Disminuye el riesgo cardiovascular",
            'magnitude': "Indica la fuerza del impacto"
        }
    }
    
    import json
    with open('models/shap_analysis.json', 'w') as f:
        json.dump(shap_analysis, f, indent=2)
    
    print("📁 Análisis SHAP guardado en 'shap_analysis.json'")
    
except Exception as e:
    print(f"❌ Error en el análisis SHAP: {str(e)}")
    print("Nota: SHAP requiere instalación adicional: pip install shap")
    import traceback
    traceback.print_exc()
