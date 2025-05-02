from machine import ADC, Timer
import array

adc = ADC(26)  # microfon pe GPIO26 / ADC0
buffer = array.array('H', [0]*3000)  # 3000 mostre
index = 0

def sample_callback(timer):
    global index
    if index < len(buffer):
        buffer[index] = adc.read_u16()
        index += 1
    else:
        timer.deinit()

# Timer hardware la 25kHz (adică la fiecare 40μs)
timer = Timer()
timer.init(freq=25000, mode=Timer.PERIODIC, callback=sample_callback)

# Așteaptă să se termine înregistrarea
while index < len(buffer):
    pass

# Scriere în fișier
with open("scor.txt", "w") as f:
    for value in buffer:
        f.write(str(value) + "\n")

print("Eșantionare completă.")
