#!/usr/bin/env python

from socket import *
from time import ctime

HOST = ''
PORT = 21567
BUFSIZ = 1024

ADDR = (HOST, PORT)

tcpSocket = socket(AF_INET, SOCK_STREAM);
tcpSocket.bind(ADDR);
tcpSocket.listen(5);

while True:
    print("waiting for connection...")
    tcpCliSock, addr = tcpSocket.accept();
    print('... connect from:', addr);

    while True:
        data = tcpCliSock.recv(BUFSIZ);
        if not data:
            break;
        tcpCliSock.send('[%s] %s' % bytes((ctime(), 'utf-8'), data))

    tcpCliSock.close();
tcpSocket.close();
