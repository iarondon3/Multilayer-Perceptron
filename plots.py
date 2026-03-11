import matplotlib.pyplot as plt
import numpy as np

def _setup_scatter(ax, inputs, colors, title):

    columns = inputs.shape[1]
    
    if columns >= 3:
        xs = inputs[:, 0]
        ys = inputs[:, 1]
        zs = inputs[:, 2]
        
        if hasattr(ax, 'zaxis'):
            ax.scatter(xs, ys, zs, c=colors, s=50, edgecolors='k')
            ax.set_zlabel('x3')
        else:
            ax.scatter(xs, ys, c=colors, s=50, edgecolors='k')
            ax.text(0.05, 0.95, "2D View (3D Data)", transform=ax.transAxes, fontsize=8, color='red')
    else:
        xs = inputs[:, 0]
        ys = inputs[:, 1]
        ax.scatter(xs, ys, c=colors, s=100, edgecolors='k')

    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, alpha=0.3)

def _add_hyperparameter_info(fig, input_neurons, output_neurons, hidden_layers, neurons_per_layer, learning_rate, epochs):

    arch_text = f"Inputs: {input_neurons} ➡ Hidden: {hidden_layers} layer(s) of {neurons_per_layer} neurons ➡ Outputs: {output_neurons}"

    training_text = f"Learning Rate: {learning_rate}  |  Total Epochs: {epochs}"

    info = f"{arch_text}\n{training_text}"

    fig.text(0.5, 0.02, info, ha='center', fontsize=10, 
             bbox=dict(facecolor='#f0f0f0', edgecolor='gray', boxstyle='round,pad=0.8'))
    
    plt.subplots_adjust(bottom=0.2)


def plot_training(inputs, targets, predictions, errors_history, input_neurons, output_neurons, hidden_layers, neurons_per_layer, learning_rate, epochs):
    is_3d = inputs.shape[1] >= 3
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d' if is_3d else None)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d' if is_3d else None)
    ax3 = fig.add_subplot(1, 3, 3)

    # Actual
    _setup_scatter(ax1, inputs, targets.flatten(), "1. Actual Data (Targets)")

    # Correct vs Incorrect
    colors = ['green' if round(p) == t else 'red' for p, t in zip(predictions.flatten(), targets.flatten())]
    _setup_scatter(ax2, inputs, colors, "2. Correct (Green) vs Incorrect (Red)")

    # 3. Error
    ax3.plot(errors_history, color='blue')
    ax3.set_title("3. Error Evolution")
    ax3.set_xlabel("Epochs")
    ax3.grid(True, alpha=0.3)

    # Hyperparameters
    _add_hyperparameter_info(fig, input_neurons, output_neurons, hidden_layers, neurons_per_layer, learning_rate, epochs)

    print("Opening plots window... (Close it to continue)")
    plt.show()

def plot_testing(inputs, targets, predictions, input_neurons, output_neurons, hidden_layers, neurons_per_layer, learning_rate, epochs):
    is_3d = inputs.shape[1] >= 3
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d' if is_3d else None)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d' if is_3d else None)
    ax3 = fig.add_subplot(1, 3, 3)

    if output_neurons > 1:
        # Multiclass
        t_indices = np.argmax(targets, axis=1)
        p_indices = np.argmax(predictions, axis=1)
    else:
        # Normal
        t_indices = targets.flatten().astype(int)
        p_indices = np.round(predictions.flatten()).astype(int)

    # Actual
    _setup_scatter(ax1, inputs, t_indices, "1. Actual Data")

    # Predictions
    hit_colors = []
    
    for actual, pred in zip(t_indices, p_indices):
        if actual == pred:
            hit_colors.append('green')
        else:
            hit_colors.append('red')
                
    _setup_scatter(ax2, inputs, hit_colors, "2. Network Predictions")


    # Confusion Matrix
    num_classes = max(output_neurons, 2) # minimum 2x2
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(t_indices, p_indices):
        confusion_matrix[t, p] += 1

    cax = ax3.matshow(confusion_matrix, cmap='Blues')
    fig.colorbar(cax, ax=ax3)


    for (i, j), z in np.ndenumerate(confusion_matrix):
        ax3.text(j, i, str(z), ha='center', va='center', color='black', fontsize=12)
    
    ax3.set_title("3. Confusion Matrix")
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')

    labels = [str(i) for i in range(num_classes)]

    ax3.set_xticks(range(num_classes))
    ax3.set_yticks(range(num_classes))
    ax3.set_xticklabels(labels)
    ax3.set_yticklabels(labels)

    # Hyperparameters
    _add_hyperparameter_info(fig, input_neurons, output_neurons, hidden_layers, neurons_per_layer, learning_rate, epochs)

    print("Opening plots window... (Close it to continue)")
    plt.show()


#     es_3d = inputs.shape[1] >= 3
#     fig = plt.figure(figsize=(16, 6))
    
#     ax1 = fig.add_subplot(1, 3, 1, projection='3d' if es_3d else None)
#     ax2 = fig.add_subplot(1, 3, 2, projection='3d' if es_3d else None)
#     ax3 = fig.add_subplot(1, 3, 3)

#     if n_out > 1:
#         #Multiclase
#         t_indices = np.argmax(targets, axis=1)
#         p_indices = np.argmax(predicciones, axis=1)
#     else:
#         #Normal
#         t_indices = targets.flatten().astype(int)
#         p_indices = np.round(predicciones.flatten()).astype(int)

#     # Reales
#     _configurar_scatter(ax1, inputs, t_indices, "1. Datos Reales")

#     # Predicciones
#     colores_acierto = []
    
#     for real, pred in zip(t_indices, p_indices):
#         if real == pred:
#             colores_acierto.append('green')
#         else:
#             colores_acierto.append('red')
                
#     _configurar_scatter(ax2, inputs, colores_acierto, "2. Predicciones de la red")


#     # Matriz Confusion
#     num_clases = max(n_out, 2) # mínimo 2x2
#     matriz = np.zeros((num_clases, num_clases), dtype=int)

#     for t, p in zip(t_indices, p_indices):
#         matriz[t, p] += 1

#     cax = ax3.matshow(matriz, cmap='Blues')
#     fig.colorbar(cax, ax=ax3)


#     for (i, j), z in np.ndenumerate(matriz):
#         ax3.text(j, i, str(z), ha='center', va='center', color='black', fontsize=12)
    
#     ax3.set_title("3. Matriz de Confusión")
#     ax3.set_xlabel('Predicho')
#     ax3.set_ylabel('Real')

#     etiquetas = [str(i) for i in range(num_clases)]

#     ax3.set_xticks(range(num_clases))
#     ax3.set_yticks(range(num_clases))
#     ax3.set_xticklabels(etiquetas)
#     ax3.set_yticklabels(etiquetas)

#     # Hiperparaetros
#     _agregar_info_hiperparametros(fig, n_in, n_out, n_ocu, n_neu, tasa, epocas)

#     print("Abriendo ventana de gráficos... (Cierrala para continuar)")
#     plt.show()