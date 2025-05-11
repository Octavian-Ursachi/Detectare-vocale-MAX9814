from machine import Pin, ADC
import time
import utime

mic = ADC(26)  # Conectați output-ul microfonului la pinul GP26 (ADC0)



led = Pin("LED", Pin.OUT)


SAMPLE_DELAY_US = 40  # 40 microsecunde între eșantioane (aprox. 25kHz)

def main():

    led.value(1)  #
    

    count = 0
    while True:
       
        sample = mic.read_u16() >> 4  # Conversia de la 16-bit la 12-bit
        
        print(sample)
        

        count += 1
        if count >= 1000:
            led.value(0)
            utime.sleep_ms(50)
            led.value(1)
            count = 0
            

        utime.sleep_us(SAMPLE_DELAY_US)

# Rulează programul principal
if __name__ == "__main__":
    main()
