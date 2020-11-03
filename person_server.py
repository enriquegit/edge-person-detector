import paho.mqtt.client as mqtt


import paho.mqtt.client as mqtt

# Before running the server mosquitto should be initialized with: mosquitto

clientid = "serverwindows1"
brokerip = "localhost"
brokerport = 1883

def on_connect(client, userdata, flags, rc):
    print("Connected result code {0}".format(str(rc)))
    client.subscribe("personCount")

def on_message(client, userdata, msg):
    print("Message received: " + msg.topic + " " + str(msg.payload))


client = mqtt.Client(clientid)
client.on_connect = on_connect
client.on_message = on_message
client.connect(brokerip, brokerport)
client.loop_forever()
