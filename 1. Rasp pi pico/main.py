from machine import ADC
import time
import select
import sys

# Configuration
SAMPLE_RATE = 25000  # Hz
NR_OF_SAMPLES = 3000
mic_adc = ADC(26) 
filename = "audio_data.txt"

def sample_microphone():
    return mic_adc.read_u16() 

def save_to_file(samples):
    with open(filename, "w") as f:
        for sample in samples:
            f.write(f"{sample}\n")
    print(f"INFO: Data saved to {filename}")

def check_for_command():
    poll = select.poll()
    poll.register(sys.stdin, select.POLLIN)
    
    if poll.poll(0):  # Non-blocking poll
        command = sys.stdin.readline().strip().upper()
        return command
    return None

def main():
    print("INFO: MAX9814 microphone recorder ready")

    while True:
        command = check_for_command()
        
        if command == "START":
            print("INFO: Recording started")
            samples = []

            # Start sampling
            for _ in range(NR_OF_SAMPLES):
                value = sample_microphone()
                samples.append(value)
                print(f"DATA:{value}")  # Send to Serial
                time.sleep(1 / SAMPLE_RATE)  # Maintain sample rate

            print("STOP")  # Indicate recording is complete
            save_to_file(samples)  # Save to file

        elif command == "EXIT":
            print("INFO: Exiting program")
            break

        time.sleep(0.1)  # Prevent CPU hogging when idle

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")

