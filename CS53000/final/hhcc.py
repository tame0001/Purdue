import time
from btlewrap import BluepyBackend

from miflora.miflora_poller import MiFloraPoller, \
    MI_CONDUCTIVITY, MI_MOISTURE, MI_LIGHT, MI_TEMPERATURE, MI_BATTERY
from miflora import miflora_scanner


address = 'C4:7C:8D:64:86:40'
# poller = MiFloraPoller(address, BluepyBackend)



reading_attemp = 5
temperature = 0
light = 0

for i in range(reading_attemp):

    poller = MiFloraPoller(address, BluepyBackend)

    reading = poller.parameter_value(MI_TEMPERATURE)
    print('{} C'.format(reading))
    temperature += reading
        
    reading = poller.parameter_value(MI_LIGHT)
    print('{} Lux'.format(reading))
    light += reading
    
    time.sleep(1)
        
temperature /= reading_attemp
light /= reading_attemp
batter = poller.parameter_value(MI_BATTERY)
print('Remaining Battery = {} %'.format(batter))
print('Temperature = {:.2f} C'.format(temperature))
print('Light Intensity = {:.2f} Lux'.format(light))