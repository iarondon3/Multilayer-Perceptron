import data
import neural_network
import plots

# Global configuration
learning_rate = 0.1

def run_training(weights, biases, previous_epochs, input_neurons, output_neurons, hidden_layers, neurons_per_layer, inputs_cache, targets_cache):
    inputs, targets = None, None

    if inputs_cache is not None:
        while True:
            response = input("Continue training with this data? (y/n): ").lower().strip()

            if response == 'y':
                inputs, targets = inputs_cache, targets_cache
                print("Using cached data.")
                break
            elif response == 'n':
                print("Load the Training file:")
                inputs, targets = data.load_data() # Fix applied here
                break
            else:
                print("Invalid option. Please type 'y' or 'n'.")

    else:
        print("Load the Training file:")
        inputs, targets = data.load_data()

    if inputs.shape[1] != input_neurons or targets.shape[1] != output_neurons:
        print(f"Error: The data has {inputs.shape[1]} inputs and {targets.shape[1]} outputs.")
        print(f"But the network expects {input_neurons} inputs and {output_neurons} outputs.")
        return weights, biases, previous_epochs, inputs_cache, targets_cache

    try:
        new_epochs = int(input("Number of epochs to train: "))
        
        weights, biases, errors = neural_network.train_model(
            inputs, targets, weights, biases, learning_rate, new_epochs, 
            start_counter=previous_epochs
        )
        
        previous_epochs += new_epochs
        print(f"Final error for epoch {previous_epochs}: {errors[-1]:.6f}")
        
        predictions, _ = neural_network.forward_propagation(inputs, weights, biases)
        
        plots.plot_training(
            inputs, targets, predictions, errors, 
            input_neurons, output_neurons, hidden_layers, neurons_per_layer, learning_rate, previous_epochs
        )

    except ValueError:
        print("Please enter a valid number of epochs.")
        
    return weights, biases, previous_epochs, inputs, targets

def run_testing(weights, biases, input_neurons, output_neurons, hidden_layers, neurons_per_layer, previous_epochs):
    print("Load the Testing file:")
    inputs_test, targets_test = data.load_data()
    
    if inputs_test.shape[1] != input_neurons:
        print(f"The file has {inputs_test.shape[1]} inputs, the network expects {input_neurons}.")
        return

    print("Testing network...")
    predictions, _ = neural_network.forward_propagation(inputs_test, weights, biases)
    
    correct_count = 0
    print("   Actual | Prediction | Result")
    print("   -------|------------|-----------")
    
    for i in range(len(targets_test)):
        actual = targets_test[i][0]
        pred = predictions[i][0]
        rounded = round(pred)
        
        match_status = "Correct" if actual == rounded else "Incorrect"
        if actual == rounded: correct_count += 1
        
        print(f"   {actual:.0f}      | {pred:.4f} ({rounded}) | {match_status}")
    
    acc = (correct_count / len(targets_test)) * 100
    print(f"Total accuracy: {acc:.2f}%")

    if acc >= 75:
        print(f"Perceptron trained with {acc:.2f}% accuracy.")
    else:
        print("More training is needed.")

    plots.plot_testing(
        inputs_test, targets_test, predictions,
        input_neurons, output_neurons, hidden_layers, neurons_per_layer, learning_rate, previous_epochs
    )


def run_save(input_neurons, output_neurons, hidden_layers, neurons_per_layer, previous_epochs, weights, biases):
    filename = input("Filename to save (e.g., saved_model.csv): ")
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    data.save_model(filename, input_neurons, output_neurons, hidden_layers, neurons_per_layer, previous_epochs, weights, biases)

if __name__ == "__main__":
    print("MULTILAYER PERCEPTRON SYSTEM")
    
    # 1. Initial Configuration
    input_neurons, output_neurons, hidden_layers, neurons_per_layer, previous_epochs, weights, biases = data.get_user_config()
    
    if weights is None:
        print("Initializing new neural network...")
        _, _, _, _, weights, biases = neural_network.initialize_parameters(input_neurons, output_neurons, hidden_layers, neurons_per_layer)
    else:
        print(f"Network restored successfully. History: {previous_epochs} epochs.")

    cached_inputs = None   
    cached_targets = None

    # 2. Main Loop
    while True:
        print(f"\nMenu (Cumulative epochs: {previous_epochs})")
        print("1. Train network")
        print("2. Test network")
        print("3. Save network")
        print("4. Exit")
        
        action = input("Select an action: ")
        
        if action == '1':
            weights, biases, previous_epochs, cached_inputs, cached_targets = run_training(
                weights, biases, previous_epochs, input_neurons, output_neurons, hidden_layers, neurons_per_layer, cached_inputs, cached_targets
            )

        elif action == '2':
            run_testing(weights, biases, input_neurons, output_neurons, hidden_layers, neurons_per_layer, previous_epochs)

        elif action == '3':
            run_save(input_neurons, output_neurons, hidden_layers, neurons_per_layer, previous_epochs, weights, biases)

        elif action == '4':
            print("Goodbye! Thank you for using the program.")
            break
        
        else:
            print("Invalid option.")

#     print("SISTEMA DE PERCEPTRÓN MULTICAPA")
    
#     # 1. Configuración Inicial
#     n_in, n_out, n_ocultas, n_neu, epocas_previas, pesos, bias = data.pedir_configuracion_usuario()
    
#     if pesos is None:
#         print("Inicializando nueva red neuronal")
#         _, _, _, _, pesos, bias = red_neuronal.inicializar_parametros(n_in, n_out, n_ocultas, n_neu)
#     else:
#         print(f"Red restaurada correctamente. Historial: {epocas_previas} épocas.")

#     inputs_memoria = None   
#     targets_memoria = None

#     # 2. Bucle Principal
#     while True:
#         print(f"\nMenu (Épocas acumuladas: {epocas_previas})")
#         print("1. Entrenar red")
#         print("2. Probar red")
#         print("3. Guardar red")
#         print("4. Salir")
        
#         accion = input("Selecciona una acción: ")
        
#         if accion == '1':
#             pesos, bias, epocas_previas, inputs_memoria, targets_memoria = ejecutar_entrenamiento(pesos, bias, epocas_previas, n_in, n_out, n_ocultas, n_neu, inputs_memoria, targets_memoria)

#         elif accion == '2':
#             ejecutar_prueba(pesos, bias, n_in)

#         elif accion == '3':
#             ejecutar_guardado(n_in, n_out, n_ocultas, n_neu, epocas_previas, pesos, bias)

#         elif accion == '4':
#             print("¡Hasta luego! Gracias por usar el programa")
#             break
        
#         else:
#             print("Opción no válida.")