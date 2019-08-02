from SocketServer import (TCPServer as TCP, StreamRequestHandler as SRH )
from time import ctime

HOST = ''
PORT = 21567
ADDR = (HOST, PORT)

class MyRequestHandler(SRH):
    def handle(self):
        data = self.rfile.readline();
        print( '...connectd from:', self.client_address, data)
        self.wfile.write('[%s] %s' % (ctime(), data))

tcpServ = TCP(ADDR, MyRequestHandler )
print('waiting for connect...')
tcpServ.serve_forever();

