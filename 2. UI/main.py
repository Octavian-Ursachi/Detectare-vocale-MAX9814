import tkinter as tk
import tkinter.ttk as tkk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np
import serial
import serial.tools
import serial.tools.list_ports
import re
import threading

BAUDRATE = 115200

class AudioRecorderUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Audio Recorder")
        self.geometry("1024x768")
        self.ser = None
        self.curr_data = []
        self.record_thread = None
        
        self.status_label = tk.Label(self, text="Status : Disconnected",fg="#f00" , font=("Arial", 14))
        self.status_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")  # Adjust alignment

        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")  # Position frame on left

        self.coms_combobox = tkk.Combobox(self.buttons_frame, width=16, height=1)
        self.coms_combobox.grid(row=0, column=0, padx=5, pady=5)

        self.refresh_button = tk.Button(self.buttons_frame, text="Refresh", width=10, height=1, command=self.refresh, font=("Arial", 14))
        self.refresh_button.grid(row=0, column=1, padx=5, pady=5)

        self.connect_button = tk.Button(self.buttons_frame, text="Connect", width=10, height=1, command=self.connect, font=("Arial", 14))
        self.connect_button.grid(row=0, column=2, padx=5, pady=5)

        self.load_button = tk.Button(self.buttons_frame, text="Load Data", width=10, height=1, command=self.load_data, font=("Arial", 14))
        self.load_button.grid(row=1, column=0, padx=5, pady=5)

        self.save_button = tk.Button(self.buttons_frame, text="Save Data", width=10, height=1, command=self.save_data, font=("Arial", 14))
        self.save_button.grid(row=1, column=1, padx=5, pady=5)

        self.record_button = tk.Button(self.buttons_frame, text="Record", width=10, height=1, command=self.record_data, font=("Arial", 14))
        self.record_button.grid(row=1, column=2, padx=5, pady=5)
        self.record_button.config(state=tk.DISABLED)

        # Canvas to display the plot
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.grid(row=10, column=0, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot([], [], color='blue', label='Audio Signal')
        self.ax.set_xlim(0, 100)  
        self.ax.set_ylim(-1, 1)  
        self.ax.set_xlabel('Sample Index')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title("Live Audio Signal")
        self.ax.grid(True)
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # Start animation
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False)

        self.init_coms()

    def record_data(self):
        if self.record_thread is None or not self.record_thread.is_alive():
            self.record_thread = threading.Thread(target=self.record_data_thread, daemon=True)
            self.record_thread.start()

    def record_data_thread(self):
        self.ser.write(b"START\n")
        self.curr_data = []

        while True:
            data = self.ser.readline().decode().strip()
            if data == "STOP":
                break

            match = re.search(r"DATA:(\d+)", data)
            if match:
                value = int(match.group(1)) 
                self.curr_data.append(value)

        self.record_thread = None

    def init_coms(self):
        ports = list(serial.tools.list_ports.comports())
        com_ports = []
        self.coms_combobox.set("Select COM port")
        for p in ports:
            match = re.search(r'\((.*?)\)', str(p))
            if match:
                result = match.group(1)
                com_ports.append(result)
        self.coms_combobox['values'] = com_ports


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

    def connect(self):
        try:
            self.ser = serial.Serial(self.coms_combobox.get(), baudrate=BAUDRATE)
            print(f"Connected to {self.coms_combobox.get()}") 
            self.status_label.config(fg="#0a0", text="Status : Connected")
            self.record_button.config(state=tk.ACTIVE)
        except Exception as e:
            print(f"Error: {e}")

    def plot_data(self, data):
        if data:
            try:
                fig, ax = plt.subplots(figsize=(10,6))

                ax.plot(data, color='blue', label='Audio Signal')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Amplitude')
                ax.set_title(f"Data")
                ax.grid(True)
                ax.legend()

                # Clear the canvas frame before placing a new plot
                for widget in self.canvas_frame.winfo_children():
                    widget.destroy()

                # Embed the plot into the Tkinter window
                self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack()  # Pack the canvas into the Tkinter frame

            except Exception as e:
                print(f"Error opening file: {e}")

    def load_data(self):
        self.curr_data = []
        filename = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if filename:
            with open(filename, "r") as f:
                for line in f:
                    try:
                        value = float(line.strip())
                        self.curr_data.append(value)
                    except ValueError:
                        continue  
        self.plot_data(self.curr_data)

    def save_data(self):
        file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if not file:
            return

        try:
            with open(file, "w") as f:
                for value in self.curr_data:
                    f.write(f"{value}\n") 
        except Exception as e:
            print(f"Error saving file: {e}")

    def update_plot(self, frame):
        """Efficiently update the plot with new data."""
        if len(self.curr_data) > 0:
            x_data = np.arange(len(self.curr_data))
            self.line.set_xdata(x_data)
            self.line.set_ydata(self.curr_data)

            self.ax.set_xlim(0, 3000)  # Fixed x-axis
            self.ax.set_ylim(0, 50000)  # Fixed y-axis

            self.canvas.draw_idle()

if __name__ == "__main__":
    app = AudioRecorderUI()
    app.mainloop()
