import tkinter as tk
import tkinter.ttk as tkk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import serial
import serial.tools
import serial.tools.list_ports
import re
import os
import collections
import threading
from scipy.spatial.distance import cosine
from scipy.signal import find_peaks  

BAUDRATE = 115200

class SignalAnalyzerApp:
    def __init__(self, root):

        self.ser = None
        self.root = root
        self.fft_data = None
        self.fft_freqs = None  
        self.sample_buffer = collections.deque(maxlen=1000)
        self.recording = False
        self.sample_size = 1000
        self.mean_signals = {"a": [], "e": [], "i": [], "o": [], "u": []}
        self.vowels = ['a', 'e', 'i', 'o', 'u']
        self.vowel_models = {}
        self.root.title("Signal Analyzer")

        self.load_vowel_models()

        # Top frame for COM selection and buttons
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # COM Port Select
        self.com_label = tk.Label(top_frame, text="Select COM Port:")
        self.com_label.pack(side=tk.LEFT)

        self.coms_combobox = tkk.Combobox(top_frame, values=["Select Com"])  # Replace with actual port scan
        self.coms_combobox.pack(side=tk.LEFT, padx=5)

        # Buttons
        self.connect_button = tk.Button(top_frame, text="Connect", command=self.connect)
        self.connect_button.pack(side=tk.LEFT, padx=5)

        self.extra_button = tk.Button(top_frame, text="Refresh", command=self.refresh)
        self.extra_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(top_frame, text="Save Signal", command=self.save_signal)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(top_frame, text="Load Signal", command=self.load_signal)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.record_button = tk.Button(top_frame, text="Record", command=self.record_audio)
        self.record_button.pack(side=tk.LEFT, padx=5)
        self.record_button.config(state=tk.DISABLED)

        self.mean_button = tk.Button(top_frame, text="Calculate Mean", command=self.calculate_mean)
        self.mean_button.pack(side=tk.LEFT, padx=5)

        # Frame for Graphs
        graph_frame = tk.Frame(root)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Signal Plot
        self.fig_signal = plt.Figure(figsize=(8, 2), dpi=100)
        self.ax_signal = self.fig_signal.add_subplot(111)
        self.ax_signal.set_title("Signal")
        self.canvas_signal = FigureCanvasTkAgg(self.fig_signal, master=self.root)
        self.canvas_signal.get_tk_widget().pack(pady=10)

        # FFT Plot
        self.fig_fft = plt.Figure(figsize=(8, 2), dpi=100)
        self.ax_fft = self.fig_fft.add_subplot(111)
        self.ax_fft.set_title("FFT")
        self.canvas_fft = FigureCanvasTkAgg(self.fig_fft, master=self.root)
        self.canvas_fft.get_tk_widget().pack(pady=10)

          # Radar chart for vowel similarity
        radar_frame = tk.LabelFrame(root, text="Vowel Similarity Radar", padx=10, pady=10)
        radar_frame.pack(padx=10, pady=10)

        self.radar_fig = Figure(figsize=(3, 3), dpi=100)
        self.radar_ax = self.radar_fig.add_subplot(111, polar=True)
        self.radar_canvas = FigureCanvasTkAgg(self.radar_fig, master=radar_frame)
        self.radar_canvas.get_tk_widget().pack()

        self.vowels = ['A', 'E', 'I', 'O', 'U']
        self.angles = np.linspace(0, 2 * np.pi, len(self.vowels), endpoint=False).tolist()
        self.angles += self.angles[:1]  # Close the loop

        # Initial empty plot
        values = [0] * len(self.vowels)
        values += values[:1]
        self.radar_line, = self.radar_ax.plot(self.angles, values, 'b-', linewidth=2)
        self.radar_fill = self.radar_ax.fill(self.angles, values, 'skyblue', alpha=0.5)

        self.radar_ax.set_xticks(self.angles[:-1])
        self.radar_ax.set_xticklabels(self.vowels)
        self.radar_ax.set_ylim(0, 1)

        # Sensor read thread
        self.reading_thread = threading.Thread(target=self.continuous_read_thread, daemon=True)
        self.reading_thread.start()

        # Label for detected vowel
        self.detected_vowel_label = tk.Label(radar_frame, text="Detected vowel: -", font=("Arial", 14))
        self.detected_vowel_label.pack(side=tk.LEFT, padx=20)

    def set_record_button_state(self, state):
        self.record_button["state"] = state

    def continuous_read_thread(self):
        while True:
            if self.ser and self.ser.is_open:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    try:
                        value = int(line)
                        self.sample_buffer.append(value)

                        if len(self.sample_buffer) > 100 and self.record_button["state"] == "disabled":
                            self.root.after(0, lambda: self.set_record_button_state("normal"))
                    except ValueError as e:
                        print(f"Parse error: {e}")
                except Exception as e:
                    print(f"Serial error: {e}")
            else:
                # Wait before retrying if serial isn't connected yet
                threading.Event().wait(0.5)

    def record_audio(self):
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("Avertisment", "Nu sunteti conecatat la pico")
            return
            
        if self.recording:
            return
            
        if len(self.sample_buffer) < 100: 
            messagebox.showwarning("Avertisment", "Nu sunt destule esantioane.")
            return
            
        self.recording = True
        self.record_button["state"] = "disabled"
        
        samples = list(self.sample_buffer)
        
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.txt")
        with open(file_path, 'w') as f:
            for sample in samples:
                f.write(f"{sample}\n")
                
        #actualizare grafice
        self.data = np.array(samples)
        self.plot_signal(self.sample_buffer)

        self.calculate_fft()
        
        # identifica vocala
        recognized_vowel = self.identify_vowel()
        self.detected_vowel_label.config(text=f"Detected vowel: {recognized_vowel.upper()}")
        
 
        # reset starea
        self.recording = False
        self.record_button["state"] = "normal"

    def plot_signal(self, buff):
        self.ax_signal.clear()
        self.ax_signal.plot(buff)
        self.ax_signal.set_title("Signal")
        self.canvas_signal.draw()

    def connect(self):
        try:
            self.ser = serial.Serial(self.coms_combobox.get(), baudrate=BAUDRATE)
            print(f"Connected to {self.coms_combobox.get()}") 
            #self.status_label.config(fg="#0a0", text="Status : Connected")
            self.record_button.config(state=tk.ACTIVE)
        except Exception as e:
            print(f"Error: {e}")

    def refresh(self):
        ports = list(serial.tools.list_ports.comports())
        com_ports = []
        self.coms_combobox.set("Select COM port")
        for p in ports:
            match = re.search(r'\((.*?)\)', str(p))
            if match:
                result = match.group(1)
                com_ports.append(result)
        self.coms_combobox['values'] = com_ports

    def save_signal(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text files", "*.txt")])
        if filename:
            with open(filename, "w") as f:
                f.write('\n'.join(str(x) for x in self.buffer))

    def load_signal(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        buff = 0
        if filename:
            with open(filename, "r") as f:
                buff = [int(line.strip()) for line in f if line.strip().isdigit()]
            self.plot_signal(buff)
            #self.calculate_fft()
            #self.compare_with_means()

    def update_vowel_radar(self, similarity_dict):
        values = [similarity_dict.get(v, 0) for v in self.vowels]
        values += values[:1]  # Close the shape

        # Update radar plot
        self.radar_line.set_ydata(values)

        # Remove old fill and add new one
        for coll in self.radar_fill:
            coll.remove()
        self.radar_fill = self.radar_ax.fill(self.angles, values, 'skyblue', alpha=0.5)

        self.radar_canvas.draw()

    def calculate_mean(self):
        vowels = ['a', 'e', 'i', 'o', 'u']
        
        for vowel in vowels:
            # Set the directory for the current vowel
            vowel_dir = os.path.join(os.getcwd(), vowel)
            
            # Check if the directory exists
            if os.path.isdir(vowel_dir):
                total_sum = np.zeros(self.sample_size)  # Initialize the sum array
                file_count = 0

                # Iterate through files in the vowel directory (e.g., a1.txt, a2.txt, etc.)
                for file_name in os.listdir(vowel_dir):
                    if file_name.endswith(".txt") and file_name.startswith(vowel):
                        file_path = os.path.join(vowel_dir, file_name)
                        with open(file_path, "r") as f:
                            # Read the signal data from the file
                            signal = [int(line.strip()) for line in f if line.strip().isdigit()]
                            if len(signal) == self.sample_size:
                                total_sum += np.array(signal)
                                file_count += 1

                if file_count > 0:
                    # Calculate the mean vector
                    mean_vector = total_sum / file_count

                    # Save the mean to the corresponding mean file (e.g., mean_vector_a.txt)
                    self.mean_signals[vowel] = mean_vector
                    mean_file_name = f"mean_vector_{vowel}.txt"
                    with open(mean_file_name, "w") as f:
                        for value in mean_vector:
                            f.write(f"{value}\n")
                    print(f"Mean signal for {vowel} saved to {mean_file_name}")
                else:
                    print(f"No valid files found for vowel '{vowel}' in the directory.")
            else:
                print(f"Directory for vowel '{vowel}' does not exist.")
        
    def calculate_fft(self):
        if self.data is None:
            return

        # normalizare semnal
        data = self.data - np.mean(self.data)
        if np.std(data) > 0:
            data = data / np.std(data)
            
        # hamming inainte de FFT ceva scurgere spectrala parca
        window = np.hamming(len(data))
        windowed_data = data * window
        
        # FFT pe datele cu fereastra aplicata
        self.fft_data = np.abs(np.fft.fft(windowed_data))
        
        freqs = np.fft.fftfreq(len(self.data), 1/25000)
        
        half_len = len(freqs) // 2
        self.fft_data = self.fft_data[:half_len]
        # --
        # self.fft_data = self.fft_data[75:half_len]
        # --

        # normalizare vector fft
        norm = np.linalg.norm(self.fft_data)
        if norm > 0:
            self.fft_data = self.fft_data / norm

        # --
        #  freqs = freqs[75:half_len]
        # --
        freqs = freqs[:half_len]

        self.fft_freqs = freqs
        
        self.ax_fft.clear()
        
        max_freq = max(freqs)
        if max_freq > 1e6:
            plot_freqs = freqs / 1e6
            freq_unit = "MHz"
        elif max_freq > 1e3:
            plot_freqs = freqs / 1e3
            freq_unit = "kHz"
        else:
            plot_freqs = freqs
            freq_unit = "Hz"
        
        self.ax_fft.plot(plot_freqs, self.fft_data)
        self.ax_fft.set_title("FFT")
        
        self.ax_fft.set_xlabel(f"Frecventa ({freq_unit})", fontsize=12)
        self.ax_fft.set_ylabel("Magnitudine")
        self.ax_fft.grid(True)
        
        # height pentru a filtra peak-urile slabe si distance pentru a evita peak-uri prea apropiate
        
        #  prag minim de detectie (30/100 din valoarea )
        height_threshold = 0.3 * np.max(self.fft_data)
        
        # Distanta minima intre peak-uri (in puncte)
        min_distance = len(self.fft_data) // 50
        
        peak_indices, _ = find_peaks(self.fft_data, height=height_threshold, distance=min_distance)
        
        max_peaks_to_show = 5
        if len(peak_indices) > max_peaks_to_show:
            sorted_peaks = sorted(peak_indices, key=lambda idx: self.fft_data[idx], reverse=True)
            peak_indices = sorted_peaks[:max_peaks_to_show]
        
        for idx in peak_indices:
            freq_value = freqs[idx]
            mag_value = self.fft_data[idx]
            
            if max_freq > 1e6:
                freq_display = f"{freq_value/1e6:.1f}"
            elif max_freq > 1e3:
                freq_display = f"{freq_value/1e3:.1f}"
            else:
                freq_display = f"{freq_value:.1f}"
                
            self.ax_fft.plot(plot_freqs[idx], mag_value, "x", color='red', markersize=5)
            self.ax_fft.annotate(
                f"{freq_display}\n{mag_value:.3f}", 
                (plot_freqs[idx], mag_value),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white")
            )
        
        self.fig_fft.subplots_adjust(bottom=0.15)
        self.canvas_fft.draw()
        
    def load_vowel_models(self):
        vowel_dirs = {
            'a': os.path.join(os.path.dirname(os.path.abspath(__file__)), "a", "mean_vector_a.txt"),
            'e': os.path.join(os.path.dirname(os.path.abspath(__file__)), "e", "mean_vector_e.txt"),
            'i': os.path.join(os.path.dirname(os.path.abspath(__file__)), "i", "mean_vector_i.txt"),
            'o': os.path.join(os.path.dirname(os.path.abspath(__file__)), "o", "mean_vector_o.txt"),
            'u': os.path.join(os.path.dirname(os.path.abspath(__file__)), "u", "mean_vector_u.txt")
        }
        
        for vowel, path in vowel_dirs.items():
            try:
                if os.path.exists(path):
                    print(f"Loading vowel model from: {path}")
                    # datele din mean vect
                    with open(path, 'r') as f:
                        values = []
                        for line in f:
                            try:
                                value = float(line.strip())
                                values.append(value)
                            except ValueError:
                                pass
                    
                    if values:
                        # normalizare semnal - imbunatatit
                        data = np.array(values[:1000])
                        # Eliminare DC offset
                        data = data - np.mean(data)
                        # Normalizare amplitude
                        if np.std(data) > 0:
                            data = data / np.std(data)
                        
                        #  FFT  doar prima jumatate
                        fft_data = np.abs(np.fft.fft(data))
                        half_len = len(fft_data) // 2
                        
                        # --
                        # fft_data = fft_data[75:half_len]
                        # --


                        fft_data = fft_data[:half_len]
                        
                        window_size = len(fft_data)
                        window = np.hamming(window_size)
                        fft_data = fft_data * window
                        
                        norm = np.linalg.norm(fft_data)
                        if norm > 0:
                            fft_data = fft_data / norm
                            
                        self.vowel_models[vowel] = fft_data

            except Exception as e:
                print(f"Eroare incarcare model {vowel}: {str(e)}")

    def identify_vowel(self):
        if self.fft_data is None or not self.vowel_models:
            return None
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        #  cosinus cu fiecare fisier individual si cu mean vector
        cosine_distances = {}
        all_file_distances = {}
        avg_distances = {}
        combined_distances = {}
        
        for vowel, model in self.vowel_models.items():
            min_length = min(len(self.fft_data), len(model))
            mean_distance = cosine(self.fft_data[:min_length], model[:min_length])
            cosine_distances[vowel] = mean_distance
            
            #  toate fisierele pentru acest vowel
            vowel_dir = os.path.join(base_dir, vowel)
            all_file_distances[vowel] = []
            
            if os.path.exists(vowel_dir):
                sample_files = [f for f in os.listdir(vowel_dir) 
                              if f.endswith('.txt') and not f.startswith('mean_vector_')]
                
                print(f"\nDistante cosinus pentru vocala '{vowel}':")
                
                for sample_file in sample_files:
                    file_path = os.path.join(vowel_dir, sample_file)
                    try:
                        # Citire date
                        with open(file_path, 'r') as f:
                            data = []
                            for line in f:
                                try:
                                    value = float(line.strip())
                                    data.append(value)
                                except ValueError:
                                    pass
                        
                        if not data:
                            continue
                        
                        data = np.array(data[:1000])
                        data = data - np.mean(data)
                        if np.std(data) > 0:
                            data = data / np.std(data)
                        
                        fft_data = np.abs(np.fft.fft(data))
                        half_len = len(fft_data) // 2
                        
                        
                        fft_data = fft_data[:half_len]
                        
                        window_size = len(fft_data)
                        window = np.hamming(window_size)
                        fft_data = fft_data * window
                        
                        # normalizare pentru comparare uniforma
                        norm = np.linalg.norm(fft_data)
                        if norm > 0:
                            fft_data = fft_data / norm
                            
                        #  distanta cosinus
                        min_len = min(len(self.fft_data), len(fft_data))
                        distance = cosine(self.fft_data[:min_len], fft_data[:min_len])
                        all_file_distances[vowel].append(distance)
                        
                        print(f"  {sample_file}: {distance:.4f} (similaritate: {1-distance:.4f})")
                        
                    except Exception as e:
                        print(f"  Eroare la procesarea fisierului {sample_file}: {str(e)}")
                
                if all_file_distances[vowel]:
                    avg_distance = sum(all_file_distances[vowel]) / len(all_file_distances[vowel])
                    avg_distances[vowel] = avg_distance
                    
                    # 40% media individuala + 60% distanta cu mean vector
                    weight_avg = 0.4
                    weight_mean = 0.6
                    combined_dist = weight_avg * avg_distance + weight_mean * mean_distance
                    combined_distances[vowel] = combined_dist
                    
                    print(f"  Media distantelor pentru vocala {vowel}: {avg_distance:.4f} (similaritate: {1-avg_distance:.4f})")
                    print(f"  Distanta cu mean_vector_{vowel}: {mean_distance:.4f} (similaritate: {1-mean_distance:.4f})")
                    print(f"  Distanta combinata: {combined_dist:.4f} (similaritate: {1-combined_dist:.4f})")
        
        if combined_distances:
            recognized_vowel = min(combined_distances.items(), key=lambda x: x[1])[0]
        else:
            recognized_vowel = min(cosine_distances.items(), key=lambda x: x[1])[0]
        
        print("\nRezultate identificare vocala finala:")
        print("Distante cosinus cu mean vectors:")
        for vowel, dist in sorted(cosine_distances.items(), key=lambda x: x[1]):
            print(f"Vocala {vowel}: {dist:.4f} (similaritate: {1-dist:.4f})")
            
        if avg_distances:
            print("\nMedia distantelor individuale:")
            for vowel, dist in sorted(avg_distances.items(), key=lambda x: x[1]):
                print(f"Vocala {vowel}: {dist:.4f} (similaritate: {1-dist:.4f})")
            
        if combined_distances:
            print("\nDistante combinate (40% media individuala + 60% mean vector):")
            for vowel, dist in sorted(combined_distances.items(), key=lambda x: x[1]):
                print(f"Vocala {vowel}: {dist:.4f} (similaritate: {1-dist:.4f})")
            
            self.update_vowel_radar({"A": 1-combined_distances['a'], "E": 1-combined_distances['e'], "I": 1-combined_distances['i'], "O": 1-combined_distances['o'], "U": 1-combined_distances['u']})

            print(f"\nVocala recunoscuta: {recognized_vowel} (similaritate: {1-combined_distances[recognized_vowel]:.4f})")
        else:
            print(f"\nVocala recunoscuta: {recognized_vowel} (similaritate: {1-cosine_distances[recognized_vowel]:.4f})")
        

        display_distances = combined_distances if combined_distances else cosine_distances
        
        return recognized_vowel


if __name__ == "__main__":
    root = tk.Tk()
    app = SignalAnalyzerApp(root)
    root.mainloop()
