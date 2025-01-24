import numpy as np
import matplotlib.pyplot as plt
import nest

# Function to chunk data into chunks of specified duration
def chunk_data(data, rate, duration_sec):
    chunk_size = int(rate * duration_sec)
    num_chunks = len(data) // chunk_size
    chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    return chunks

# Function to plot data
def plot_data(data, title):
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

# Function to plot raster plot
def raster_plot(raster_data, chunk_idx, num_total_neurons):
    plt.figure(figsize=(10, 5))
    plt.eventplot(raster_data, colors='black')
    plt.title(f'Raster Plot for Chunk {chunk_idx + 1}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.xlim(0, 50)  # Limit x-axis to 50 ms
    plt.ylim(0, num_total_neurons)
    plt.show()

def get_current_for_idx(idx):
    # Load the current_spikes_values from npy file
    current_spikes_values = np.load('results/current_spikes_values.npy')

    # Check if the index is within the valid range
    if 0 <= idx < len(current_spikes_values):
        return current_spikes_values[idx][0]
    else:
        raise ValueError("Index out of range")

# Function to get currents for a given value
def get_currents_for_value(value):
    # Create an array to store currents for each neuron
    currents = np.zeros(6)

    # Normalize the value to be within the range of -99999 to 99999
    value = max(-99999, min(value, 99999))

    # Compute currents for each neuron
    if value >= 0:
        currents[5] = get_current_for_idx(0)  # Neuron 5: No spike for positive values
    else:
        currents[5] = get_current_for_idx(1)  # Neuron 5: Activate negative spike neuron
    currents[0] = get_current_for_idx(value % 10)  # Neuron 0: Increment of 1
    currents[1] = get_current_for_idx((value // 10) % 10)  # Neuron 1: Increment of 10
    currents[2] = get_current_for_idx((value // 100) % 10)  # Neuron 2: Increment of 100
    currents[3] = get_current_for_idx((value // 1000) % 10)  # Neuron 3: Increment of 1000
    currents[4] = get_current_for_idx((value // 10000) % 10)  # Neuron 4: Increment of 10000
    return currents

# Function to simulate SNN and plot results
def simulate_snn(chunks, num_total_neurons):
    nest.ResetKernel()
    nest.set_verbosity(20)  # Set NEST verbosity level to 20
    nest.SetKernelStatus({'print_time': False})
    neuron_model = 'iaf_psc_alpha'
    neurons = nest.Create(neuron_model, num_total_neurons, params=neuron_params)
    spike_detector = nest.Create('spike_recorder')

    nest.Connect(neurons, spike_detector)

    num_chunks = len(chunks)
    for chunk_idx, chunk in enumerate(chunks):
        print("Simulating chunk", (chunk_idx+1), "of",num_chunks,"...")
        neuron_idx = 0
        raster_data = []  # List to store spike times for raster plot

        # Reset neuron parameters and spike recorder
        nest.SetStatus(neurons, neuron_params)

        for sample in chunk:
            # Convert amplitude to current and scale to suitable range
            currents_left = get_currents_for_value(sample[0])
            currents_right = get_currents_for_value(sample[1])

            # Set the current to 6 neurons of the left channel
            for i in range(0, 6):
                nest.SetStatus(neurons[neuron_idx], {'I_e': currents_left[i]})
                neuron_idx += 1

            # Set the current to 6 neurons of the right channel
            for i in range(0, 6):
                nest.SetStatus(neurons[neuron_idx], {'I_e': currents_right[i]})
                neuron_idx += 1

        nest.Simulate(50.0)  # Simulate for 50 ms

    # Get spike times
    events = spike_detector.get("events")
    senders = events["senders"]
    ts = events["times"]
    # Plot raster plot
    plt.figure(figsize=(10, 6))
    plt.vlines(ts, senders, senders + 1, color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Raster Plot')
    plt.grid()
    plt.show()


neuron_params = {
    'C_m': 250.0,       # Membrane capacitance (pF)
    'tau_m': 10.0,      # Membrane time constant (ms)
    't_ref': 2.0,       # Refractory period (ms)
    'E_L': 0.0,         # Resting membrane potential (mV)
    'V_th': 20.0,       # Threshold potential (mV)
    'V_reset': 10.0,    # Reset potential (mV)
    'tau_syn_ex': 0.5,  # Excitatory synaptic time constant (ms)
    'tau_syn_in': 0.5   # Inhibitory synaptic time constant (ms)
}

# Load the data from npy files
data = np.load('results/data.npy')
rate = np.load('results/rate.npy')

# Chunk the data into chunks of 1/30 second
chunk_duration = 1 / 30  # seconds
chunks = chunk_data(data, rate, chunk_duration)

num_samples = len(chunks[0])
num_neurons_per_sample = 12  # 6 neurons per channel, 2 channels

num_total_neurons = num_samples * (num_neurons_per_sample + 1)  # Additional neuron for negative spike

print("\nSimulating SNN for Sound chunks:")
simulate_snn(chunks, num_total_neurons)
