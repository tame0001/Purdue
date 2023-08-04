import paho.mqtt.client as mqtt
import netifaces as ni
import json
import schedule

def find_ip():
    ip_table = []
    interfaces = ni.interfaces()
    for interface in interfaces:
        ip_address = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']
        # print(interface)
        # print(ip_address)
        ip_templete = {
            'name': interface,
            'address': ip_address
        }
        ip_table.append(ip_templete)
    print(ip_table)
    return ip_table

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.publish('me588g4/ipaddress', 'Raspberry Pi Online')
    
def send_ip(mqttc):
    ip_table = find_ip()
    mqttc.publish('me588g4/ipaddress', json.dumps(ip_table))

client = mqtt.Client()
client.on_connect = on_connect
client.connect("iot.eclipse.org", 1883, 60)
client.loop_start()

schedule.every(60).seconds.do(send_ip, client)
while True:
    schedule.run_pending()