import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("🤖 ENTRENAMIENTO DE MODELOS DE MACHINE LEARNING")
print("=" * 55)
print("Universidad Privada Antenor Orrego")
print("Predicción de Riesgo Cardiovascular")
print("=" * 55)

try:
    # Cargar datos preprocesados
    print("\n📊 Cargando datos preprocesados...")
    
    X_train = np.load('models/X_train.npy')
    X_val = np.load('models/X_val.npy')
    X_test = np.load('models/X_test.npy')
    y_train = np.load('models/y_train.npy')
    y_val = np.load('models/y_val.npy')
    y_test = np.load('models/y_test.npy')
    
    # Cargar nombres de características
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"✅ Datos cargados exitosamente")
    print(f"Entrenamiento: {X_train.shape[0]:,} registros")
    print(f"Validación: {X_val.shape[0]:,} registros")
    print(f"Prueba: {X_test.shape[0]:,} registros")
    print(f"Características: {X_train.shape[1]}")
    
    # Diccionario para almacenar resultados
    results = {}
    
    print("\n" + "="*60)
    print("🌳 MODELO 1: RANDOM FOREST")
    print("="*60)
    
    # Random Forest
    print("Entrenando Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predicciones
    rf_train_pred = rf_model.predict(X_train)
    rf_val_pred = rf_model.predict(X_val)
    rf_test_pred = rf_model.predict(X_test)
    
    rf_train_proba = rf_model.predict_proba(X_train)[:, 1]
    rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
    rf_test_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Métricas
    rf_train_auc = roc_auc_score(y_train, rf_train_proba)
    rf_val_auc = roc_auc_score(y_val, rf_val_proba)
    rf_test_auc = roc_auc_score(y_test, rf_test_proba)
    
    print(f"📊 Resultados Random Forest:")
    print(f"   AUC Entrenamiento: {rf_train_auc:.4f}")
    print(f"   AUC Validación: {rf_val_auc:.4f}")
    print(f"   AUC Prueba: {rf_test_auc:.4f}")
    
    # Importancia de características
    feature_importance_rf = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 características más importantes (Random Forest):")
    for i, (_, row) in enumerate(feature_importance_rf.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<15}: {row['importance']:.4f}")
    
    # Guardar modelo
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    
    results['random_forest'] = {
        'train_auc': rf_train_auc,
        'val_auc': rf_val_auc,
        'test_auc': rf_test_auc,
        'feature_importance': feature_importance_rf.to_dict('records')
    }
    
    print("\n" + "="*60)
    print("🚀 MODELO 2: XGBOOST")
    print("="*60)
    
    # XGBoost
    print("Entrenando XGBoost...")
    
    # Calcular scale_pos_weight para balancear clases
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    
    # Entrenar con early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Predicciones
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_test_pred = xgb_model.predict(X_test)
    
    xgb_train_proba = xgb_model.predict_proba(X_train)[:, 1]
    xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Métricas
    xgb_train_auc = roc_auc_score(y_train, xgb_train_proba)
    xgb_val_auc = roc_auc_score(y_val, xgb_val_proba)
    xgb_test_auc = roc_auc_score(y_test, xgb_test_proba)
    
    print(f"📊 Resultados XGBoost:")
    print(f"   AUC Entrenamiento: {xgb_train_auc:.4f}")
    print(f"   AUC Validación: {xgb_val_auc:.4f}")
    print(f"   AUC Prueba: {xgb_test_auc:.4f}")
    
    # Importancia de características
    feature_importance_xgb = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 características más importantes (XGBoost):")
    for i, (_, row) in enumerate(feature_importance_xgb.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<15}: {row['importance']:.4f}")
    
    # Guardar modelo
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    
    results['xgboost'] = {
        'train_auc': xgb_train_auc,
        'val_auc': xgb_val_auc,
        'test_auc': xgb_test_auc,
        'feature_importance': feature_importance_xgb.to_dict('records')
    }
    
    print("\n" + "="*60)
    print("🧠 MODELO 3: RED NEURONAL")
    print("="*60)
    
    # Red Neuronal
    print("Entrenando Red Neuronal...")
    
    nn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    nn_model.fit(X_train, y_train)
    
    # Predicciones
    nn_train_pred = nn_model.predict(X_train)
    nn_val_pred = nn_model.predict(X_val)
    nn_test_pred = nn_model.predict(X_test)
    
    nn_train_proba = nn_model.predict_proba(X_train)[:, 1]
    nn_val_proba = nn_model.predict_proba(X_val)[:, 1]
    nn_test_proba = nn_model.predict_proba(X_test)[:, 1]
    
    # Métricas
    nn_train_auc = roc_auc_score(y_train, nn_train_proba)
    nn_val_auc = roc_auc_score(y_val, nn_val_proba)
    nn_test_auc = roc_auc_score(y_test, nn_test_proba)
    
    print(f"📊 Resultados Red Neuronal:")
    print(f"   AUC Entrenamiento: {nn_train_auc:.4f}")
    print(f"   AUC Validación: {nn_val_auc:.4f}")
    print(f"   AUC Prueba: {nn_test_auc:.4f}")
    print(f"   Iteraciones: {nn_model.n_iter_}")
    
    # Guardar modelo
    joblib.dump(nn_model, 'models/neural_network_model.pkl')
    
    results['neural_network'] = {
        'train_auc': nn_train_auc,
        'val_auc': nn_val_auc,
        'test_auc': nn_test_auc,
        'iterations': int(nn_model.n_iter_)
    }
    
    print("\n" + "="*60)
    print("📊 COMPARACIÓN DE MODELOS")
    print("="*60)
    
    # Comparar modelos
    comparison_df = pd.DataFrame({
        'Modelo': ['Random Forest', 'XGBoost', 'Red Neuronal'],
        'AUC Entrenamiento': [rf_train_auc, xgb_train_auc, nn_train_auc],
        'AUC Validación': [rf_val_auc, xgb_val_auc, nn_val_auc],
        'AUC Prueba': [rf_test_auc, xgb_test_auc, nn_test_auc]
    })
    
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Determinar mejor modelo
    best_model_idx = comparison_df['AUC Validación'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Modelo']
    best_auc = comparison_df.loc[best_model_idx, 'AUC Validación']
    
    print(f"\n🏆 MEJOR MODELO: {best_model_name}")
    print(f"   AUC en Validación: {best_auc:.4f}")
    
    # Guardar el mejor modelo como modelo principal
    if best_model_name == 'Random Forest':
        joblib.dump(rf_model, 'models/best_model.pkl')
        best_model_type = 'random_forest'
    elif best_model_name == 'XGBoost':
        joblib.dump(xgb_model, 'models/best_model.pkl')
        best_model_type = 'xgboost'
    else:
        joblib.dump(nn_model, 'models/best_model.pkl')
        best_model_type = 'neural_network'
    
    # Guardar resultados completos
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'best_model': best_model_name,
        'best_model_type': best_model_type,
        'best_auc': float(best_auc),
        'models': results,
        'comparison': comparison_df.to_dict('records')
    }
    
    with open('models/training_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"📁 Modelos guardados en la carpeta 'models/'")
    print(f"🎯 Mejor modelo: {best_model_name} (AUC: {best_auc:.4f})")
    print(f"📊 Resultados detallados guardados en 'training_results.json'")
    
except Exception as e:
    print(f"❌ Error durante el entrenamiento: {str(e)}")
    import traceback
    traceback.print_exc()
