from paho.mqtt import client as mqtt_client
import time
import random
from datetime import datetime
import json

# MQTT setup requirements
broker = 'private-server.uk.to'
port = 1883
topic = "jason/pose"
topic2 = "robot/docking"
client_id = 'python-mqtt-{}'.format(random.randint(0, 1000))
username = 'user'
password = 'user'

dock = ""

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client
    
def publish(client, pose, loca1):
    #print("hello")
    # JSON format
    msgJson = {
        "pose" : pose,
        "posisi" : loca1,
    }
    
    msgString = json.dumps(msgJson)
    result = client.publish(topic, msgString)
    #print(result[0])
    
    status = result[0]
    if status == 0:
        print("Send {} to topic: {}". format(msgString, topic))
    else:
        print("Failed to send message to topic: {}". format(msgString, topic))
        
def subscribe(client: mqtt_client, topic2):
    def on_message(client, userdata, msg):
        global dock
        #print("Received {} from topic: {}".format(msg.payload.decode(), msg.topic))
        dock = msg.payload.decode()
    client.subscribe(topic2)
    client.on_message = on_message
    
def run():
    client = connect_mqtt()
    client.loop_start()
    while True:
        time.sleep(1)
        publish(client, random.randint(0, 1000), 'A')
        
def start():
    client = connect_mqtt()
    client.loop_start()
    return client

if __name__ == "__main__":
    run()
        