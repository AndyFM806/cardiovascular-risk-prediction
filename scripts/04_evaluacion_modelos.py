import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import joblib
import json

print("📊 EVALUACIÓN DETALLADA DE MODELOS")
print("=" * 45)
print("Universidad Privada Antenor Orrego")
print("=" * 45)

try:
    # Cargar datos de prueba
    print("\n📂 Cargando datos de prueba...")
    X_test = np.load('models/X_test.npy')
    y_test = np.load('models/y_test.npy')
    
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"✅ Datos cargados: {X_test.shape[0]:,} registros de prueba")
    
    # Cargar modelos
    print("\n🤖 Cargando modelos entrenados...")
    rf_model = joblib.load('models/random_forest_model.pkl')
    xgb_model = joblib.load('models/xgboost_model.pkl')
    nn_model = joblib.load('models/neural_network_model.pkl')
    
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'Red Neuronal': nn_model
    }
    
    print("✅ Modelos cargados exitosamente")
    
    # Configurar matplotlib para español
    plt.rcParams['font.size'] = 10
    sns.set_style("whitegrid")
    
    print("\n" + "="*60)
    print("📈 EVALUACIÓN EN CONJUNTO DE PRUEBA")
    print("="*60)
    
    # Diccionario para almacenar métricas
    evaluation_results = {}
    
    # Evaluar cada modelo
    for model_name, model in models.items():
        print(f"\n🔍 Evaluando {model_name}...")
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"   📊 Métricas de {model_name}:")
        print(f"      Exactitud (Accuracy): {accuracy:.4f}")
        print(f"      Precisión: {precision:.4f}")
        print(f"      Sensibilidad (Recall): {recall:.4f}")
        print(f"      F1-Score: {f1:.4f}")
        print(f"      AUC-ROC: {auc:.4f}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"   📋 Matriz de Confusión:")
        print(f"      Verdaderos Negativos: {tn:,}")
        print(f"      Falsos Positivos: {fp:,}")
        print(f"      Falsos Negativos: {fn:,}")
        print(f"      Verdaderos Positivos: {tp:,}")
        
        # Especificidad y sensibilidad
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        print(f"   🎯 Métricas Clínicas:")
        print(f"      Sensibilidad: {sensitivity:.4f} ({sensitivity*100:.1f}%)")
        print(f"      Especificidad: {specificity:.4f} ({specificity*100:.1f}%)")
        
        # Guardar resultados
        evaluation_results[model_name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        }
    
    print("\n" + "="*60)
    print("🏆 COMPARACIÓN FINAL DE MODELOS")
    print("="*60)
    
    # Crear tabla comparativa
    comparison_metrics = []
    for model_name, metrics in evaluation_results.items():
        comparison_metrics.append({
            'Modelo': model_name,
            'Exactitud': f"{metrics['accuracy']:.4f}",
            'Precisión': f"{metrics['precision']:.4f}",
            'Sensibilidad': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'AUC-ROC': f"{metrics['auc_roc']:.4f}",
            'Especificidad': f"{metrics['specificity']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_metrics)
    print(comparison_df.to_string(index=False))
    
    # Determinar el mejor modelo basado en AUC
    best_model_name = max(evaluation_results.keys(), 
                         key=lambda x: evaluation_results[x]['auc_roc'])
    best_auc = evaluation_results[best_model_name]['auc_roc']
    
    print(f"\n🥇 MODELO GANADOR: {best_model_name}")
    print(f"   AUC-ROC: {best_auc:.4f}")
    print(f"   Interpretación: {'Excelente' if best_auc > 0.9 else 'Muy Bueno' if best_auc > 0.8 else 'Bueno' if best_auc > 0.7 else 'Regular'}")
    
    print("\n" + "="*60)
    print("🎯 INTERPRETACIÓN CLÍNICA")
    print("="*60)
    
    best_metrics = evaluation_results[best_model_name]
    
    print(f"Para el modelo {best_model_name}:")
    print(f"")
    print(f"🔍 SENSIBILIDAD: {best_metrics['sensitivity']:.1%}")
    print(f"   → De cada 100 pacientes CON enfermedad cardiovascular,")
    print(f"     el modelo identifica correctamente a {best_metrics['sensitivity']*100:.0f}")
    print(f"")
    print(f"🔍 ESPECIFICIDAD: {best_metrics['specificity']:.1%}")
    print(f"   → De cada 100 pacientes SIN enfermedad cardiovascular,")
    print(f"     el modelo identifica correctamente a {best_metrics['specificity']*100:.0f}")
    print(f"")
    print(f"🔍 PRECISIÓN: {best_metrics['precision']:.1%}")
    print(f"   → De cada 100 predicciones POSITIVAS del modelo,")
    print(f"     {best_metrics['precision']*100:.0f} son realmente casos positivos")
    print(f"")
    print(f"🔍 AUC-ROC: {best_metrics['auc_roc']:.3f}")
    print(f"   → Probabilidad de que el modelo clasifique correctamente")
    print(f"     un caso positivo vs uno negativo")
    
    print("\n" + "="*60)
    print("💡 RECOMENDACIONES CLÍNICAS")
    print("="*60)
    
    sensitivity = best_metrics['sensitivity']
    specificity = best_metrics['specificity']
    
    if sensitivity >= 0.85 and specificity >= 0.85:
        print("✅ EXCELENTE RENDIMIENTO:")
        print("   • El modelo es adecuado para screening y diagnóstico")
        print("   • Bajo riesgo de falsos negativos y falsos positivos")
        print("   • Recomendado para uso clínico con supervisión médica")
    elif sensitivity >= 0.80:
        print("✅ BUEN RENDIMIENTO PARA SCREENING:")
        print("   • El modelo es útil para identificar pacientes de riesgo")
        print("   • Recomendado como herramienta de apoyo diagnóstico")
        print("   • Requiere confirmación con evaluación clínica completa")
    else:
        print("⚠️ RENDIMIENTO MODERADO:")
        print("   • El modelo puede usarse como herramienta complementaria")
        print("   • No recomendado como única herramienta diagnóstica")
        print("   • Requiere mejoras antes del uso clínico")
    
    print(f"\n📋 CASOS DE USO RECOMENDADOS:")
    print(f"   • Screening poblacional de riesgo cardiovascular")
    print(f"   • Priorización de pacientes para evaluación cardiológica")
    print(f"   • Apoyo en decisiones de medicina preventiva")
    print(f"   • Identificación temprana de factores de riesgo")
    
    # Guardar evaluación completa
    final_evaluation = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'test_set_size': int(len(y_test)),
        'best_model': best_model_name,
        'models_evaluation': evaluation_results,
        'comparison_table': comparison_df.to_dict('records'),
        'clinical_interpretation': {
            'sensitivity_interpretation': f"Identifica {best_metrics['sensitivity']*100:.0f}% de casos positivos",
            'specificity_interpretation': f"Identifica {best_metrics['specificity']*100:.0f}% de casos negativos",
            'precision_interpretation': f"{best_metrics['precision']*100:.0f}% de predicciones positivas son correctas",
            'auc_interpretation': f"Rendimiento {'excelente' if best_auc > 0.9 else 'muy bueno' if best_auc > 0.8 else 'bueno'}"
        }
    }
    
    with open('models/evaluation_results.json', 'w') as f:
        json.dump(final_evaluation, f, indent=2)
    
    print(f"\n✅ EVALUACIÓN COMPLETADA")
    print(f"📊 Resultados detallados guardados en 'evaluation_results.json'")
    print(f"🎯 Modelo recomendado: {best_model_name}")
    
except Exception as e:
    print(f"❌ Error durante la evaluación: {str(e)}")
    import traceback
    traceback.print_exc()
