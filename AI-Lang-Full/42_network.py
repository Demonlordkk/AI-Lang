"""Phase 42: networking abstraction."""
import socket
class Network:
    @staticmethod
    def tcp_connect(host, port, timeout=10):
        return socket.create_connection((host, port), timeout=timeout)
