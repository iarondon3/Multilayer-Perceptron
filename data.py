import numpy as np
import csv

def load_data(): 

    dataset = []

    while True:
        try:
            user_path = input("Please enter the full file path: ")
            file_path = user_path.strip('"').strip("'")

            with open(file_path, 'r', encoding='utf-8-sig') as file:

                headers = next(file).strip().split(',')
                print(f"Headers found: {headers}")

                for line in file:
                    clean_line = line.strip()
                    if not clean_line:
                        continue

                    text_values = clean_line.split(',')

                    numeric_row = []
                
                    for value in text_values:
                        clean_value = value.strip()
                        if clean_value:
                            numeric_row.append(float(clean_value))

                    if numeric_row:
                        dataset.append(numeric_row)
            break

        except FileNotFoundError:
            print("File not found. Please check the path and try again.")

        except Exception as e:
            print(f"An error occurred while reading the file: {e}")

    data_matrix = np.array(dataset)

    inputs = data_matrix[:, :-1]
    targets = data_matrix[:, -1]

    unique_classes = np.unique(targets)

    if len(unique_classes) > 2:
        print(f"Multiclass detected ({len(unique_classes)} classes). Transforming outputs...")
        
        # One-Hot Encoding
        n_rows = len(targets)
        n_classes = len(unique_classes)
        targets_one_hot = np.zeros((n_rows, n_classes))

        class_map = {val: i for i, val in enumerate(unique_classes)}
        
        for i, value in enumerate(targets):
            index = class_map[value]
            targets_one_hot[i, index] = 1.0
            
        targets = targets_one_hot
        
    else:
        # Original 
        targets = targets.reshape(-1, 1)
        targets = (targets + 1) / 2 # NEW TEST ROW


    max_vals = np.amax(inputs, axis=0)
    max_vals[max_vals == 0] = 1

    inputs = inputs / max_vals

    print(f"Data loaded successfully. Input matrix shape: {inputs.shape}")

    return inputs, targets

def save_model(filename, input_neurons, output_neurons, hidden_layers, neurons_per_layer, current_epochs, weights, biases):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        writer.writerow(['u', input_neurons])
        writer.writerow(['v', output_neurons])
        writer.writerow(['L', hidden_layers]) 
        writer.writerow(['b', neurons_per_layer])
        writer.writerow(['e', current_epochs])

        for i, (w_matrix, b_vector) in enumerate(zip(weights, biases)):
            # Determine layer type and number
            is_last_layer = (i == len(weights) - 1)
            layer_type = 'o' if is_last_layer else 'h' # 'o' output , 'h' hidden
            layer_num = i + 1

            neurons_in_layer = w_matrix.shape[1]

            for neuron_idx in range(neurons_in_layer):
                neuron_weights = w_matrix[:, neuron_idx].tolist()
                bias_value = b_vector[0, neuron_idx]

                row = [layer_type, layer_num, neuron_idx] + neuron_weights + [bias_value]
                writer.writerow(row)

    print(f"Network saved successfully in: {filename}")

def load_model():
    while True:
        try:
            file_path = input("Enter the PERCEPTRON file path (.csv): ").strip('"').strip("'")
            
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                rows = list(reader)

            input_neurons = int(rows[0][1])
            output_neurons = int(rows[1][1])
            hidden_layers = int(rows[2][1])
            neurons_per_layer = int(rows[3][1])
            previous_epochs = int(rows[4][1])
            
            print(f"File loaded. Configuration detected:")
            print(f"   Inputs: {input_neurons} | Outputs: {output_neurons}")
            print(f"   Hidden Layers: {hidden_layers} | Neurons/layer: {neurons_per_layer}")
            print(f"   Trained for: {previous_epochs} epochs")

            dimensions = [input_neurons] + [neurons_per_layer] * hidden_layers + [output_neurons]

            weights = []
            biases = []

            for i in range(len(dimensions) - 1):
                layer_inputs = dimensions[i]
                layer_neurons = dimensions[i+1]
                weights.append(np.zeros((layer_inputs, layer_neurons)))
                biases.append(np.zeros((1, layer_neurons)))

            for line in rows[5:]:
                
                if not line or len(line) < 3:
                    continue 

                layer_type = line[0]       
                layer_idx = int(line[1]) - 1
                neuron_idx = int(line[2])

                values = [float(x) for x in line[3:]]
            
                my_weights = values[:-1] 
                my_bias = values[-1]

                weights[layer_idx][:, neuron_idx] = my_weights
                biases[layer_idx][0, neuron_idx] = my_bias
            
            return input_neurons, output_neurons, hidden_layers, neurons_per_layer, previous_epochs, weights, biases
    
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"Error reading the file: {e}")
            return None

def get_user_config():
    print("Let's start the configuration")
    print('1. Create a new multilayer perceptron')
    print('2. Load a saved model')

    while True:

        option = input('Select an option: ')


        if option == '1':
            try:
                print('Enter the perceptron configuration')
                input_neurons = int(input('Number of input neurons: '))
                output_neurons = int(input('Number of output neurons: '))
                hidden_layers = int(input('Number of hidden layers: '))
                neurons_per_layer = int(input('Number of neurons per layer: '))

                return input_neurons, output_neurons, hidden_layers, neurons_per_layer, 0, None, None
            
            except ValueError:
                print("Please enter only integer numbers.")

    
        elif option == '2':
            result = load_model()
            if result is not None:
                return result

        else:
            print("Please enter a valid option: 1 or 2")
