import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from datetime import datetime

def load_metrics(model_dir):
    """Carga las métricas de un directorio de resultados del modelo"""
    metrics = {}
    metrics_path = os.path.join(model_dir, 'metrics_report.txt')
    
    if not os.path.exists(metrics_path):
        print(f"No se encontró el archivo de métricas en {model_dir}")
        return None
    
    with open(metrics_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'ROC AUC Score:' in line:
                metrics['auc'] = float(line.split(':')[-1].strip())
            elif 'F1 Score:' in line:
                metrics['f1'] = float(line.split(':')[-1].strip())
    
    # Cargar historial de entrenamiento
    history_path = os.path.join(model_dir, 'training_history.png')
    if os.path.exists(history_path):
        metrics['history_plot'] = history_path
    
    # Cargar matriz de confusión
    cm_path = os.path.join(model_dir, 'confusion_matrix.png')
    if os.path.exists(cm_path):
        metrics['confusion_matrix'] = cm_path
    
    # Cargar curva ROC
    roc_path = os.path.join(model_dir, 'roc_curve.png')
    if os.path.exists(roc_path):
        metrics['roc_curve'] = roc_path
    
    return metrics

def load_latest_results():
    """Carga los resultados más recientes de CNN y MOE"""
    # Directorio base del proyecto
    base_dir = os.path.abspath('.')
    results = {}
    
    # Buscar resultados más recientes de CNN (en la raíz)
    cnn_dirs = sorted(glob(os.path.join(base_dir, 'run_*')), reverse=True)
    if cnn_dirs:
        results['cnn'] = {
            'path': cnn_dirs[0],
            'metrics': load_metrics(cnn_dirs[0])
        }
    
    # Buscar resultados más recientes de MOE (en carpeta moe/)
    moe_dirs = sorted(glob(os.path.join(base_dir, 'moe', 'run_*')), reverse=True)
    if moe_dirs:
        results['moe'] = {
            'path': moe_dirs[0],
            'metrics': load_metrics(moe_dirs[0])
        }
    
    return results

def plot_comparison(metrics_dict, output_dir='mlp2/comparisons'):
    """Genera gráficos comparativos entre los modelos"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear figura para métricas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Datos para las barras
    models = list(metrics_dict.keys())
    auc_scores = [metrics_dict[m]['metrics']['auc'] for m in models]
    f1_scores = [metrics_dict[m]['metrics']['f1'] for m in models]
    
    # Gráfico de AUC
    bars1 = ax1.bar(models, auc_scores, color=['#1f77b4', '#ff7f0e'])
    ax1.set_title('Comparación de AUC')
    ax1.set_ylim(0, 1.0)
    ax1.bar_label(bars1, fmt='%.3f')
    
    # Gráfico de F1
    bars2 = ax2.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e'])
    ax2.set_title('Comparación de F1-Score')
    ax2.set_ylim(0, 1.0)
    ax2.bar_label(bars2, fmt='%.3f')
    
    plt.tight_layout()
    
    # Guardar figura
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'model_comparison_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()
    
    # Crear reporte de comparación
    report_path = os.path.join(output_dir, f'comparison_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("=== Comparación de Modelos ===\n\n")
        
        for model, data in metrics_dict.items():
            f.write(f"Modelo: {model.upper()}\n")
            f.write(f"Ruta: {data['path']}\n")
            f.write("Métricas:\n")
            f.write(f"  - AUC: {data['metrics'].get('auc', 'N/A'):.4f}\n")
            f.write(f"  - F1-Score: {data['metrics'].get('f1', 'N/A'):.4f}\n\n")
        
        # Determinar el mejor modelo basado en AUC
        best_model = max(metrics_dict.items(), 
                        key=lambda x: x[1]['metrics'].get('auc', 0))
        f.write(f"\nMejor modelo según AUC: {best_model[0].upper()} "
               f"(AUC: {best_model[1]['metrics'].get('auc', 0):.4f})\n")
    
    print(f"\nComparación guardada en: {os.path.abspath(output_dir)}")
    print(f"- Gráficos: {os.path.basename(output_path)}")
    print(f"- Reporte: {os.path.basename(report_path)}")

def main():
    print("Cargando resultados de los modelos...")
    results = load_latest_results()
    
    if not results:
        print("No se encontraron resultados de modelos para comparar.")
        return
    
    print("\nModelos encontrados:")
    for model, data in results.items():
        print(f"- {model.upper()}: {data['path']}")
    
    print("\nGenerando comparativa...")
    plot_comparison(results)

if __name__ == "__main__":
    main()
