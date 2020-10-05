import socket

HOST = '127.0.0.1'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print('Server initialized on',(HOST,PORT))
    conn, addr = s.accept()
    with conn:
        print('Client Connection', addr)
        while True:
            data = s.recv()
            print(data)