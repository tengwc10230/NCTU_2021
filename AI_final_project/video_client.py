from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time

arg = argparse.ArgumentParser()
arg.add_argument("-i", "--server-ip", required=True, help="ip address of the server")
arg.add_argument("-p", "--server-port", required=True, help="port address of the server")
args = vars(arg.parse_args())

sender = imagezmq.ImageSender(connect_to="tcp://{}:{}".format(args['server_ip'], args['server_port']))
client_name = socket.gethostname()
vs = VideoStream(src=0).start()
time.sleep(1)

while True:
    frame = vs.read()
    sender.send_image(client_name, frame)