import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('72.20.10.10',80))
if result == 0:
   print ("Port is open")
else:
   print ("Port is not open")
sock.close()