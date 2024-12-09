# client.py
import cv2
import socket
import numpy as np
import threading
import queue
import msgpack
import msgpack_numpy as m
m.patch()

class CameraClient:
    def __init__(self, server_host='100.91.37.49', server_port=9999):
        self.server_host = server_host
        self.server_port = server_port
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 解像度を設定
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.processed_frame_queue = queue.Queue(maxsize=2)
        self.running = True
        self.prev_frame = None

    def connect(self):
        try:
            self.client_socket.connect((self.server_host, self.server_port))
            self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Connected to server: {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def receive_processed_frames(self):
        buffer = bytearray()
        while self.running:
            try:
                # メッセージサイズの受信
                size_data = self.client_socket.recv(4)
                if not size_data:
                    break

                msg_size = int.from_bytes(size_data, 'big')

                # データの受信
                while len(buffer) < msg_size:
                    chunk = self.client_socket.recv(min(msg_size - len(buffer), 8192))
                    if not chunk:
                        return
                    buffer.extend(chunk)

                # データの解凍とデコード
                msg_data = buffer[:msg_size]
                buffer = buffer[msg_size:]

                frame = msgpack.unpackb(msg_data)
                frame = np.asarray(frame)

                if self.processed_frame_queue.full():
                    self.processed_frame_queue.get()
                self.processed_frame_queue.put(frame)

            except Exception as e:
                print(f"Error receiving frame: {e}")
                break

    def run(self):
        if not self.connect():
            return

        receive_thread = threading.Thread(target=self.receive_processed_frames)
        receive_thread.start()

        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    break

                # フレームをリサイズして圧縮
                frame = cv2.resize(frame, (640, 480))

                # フレームの圧縮と送信
                frame_data = msgpack.packb(frame)
                size_bytes = len(frame_data).to_bytes(4, 'big')
                self.client_socket.sendall(size_bytes + frame_data)

                # 処理済みフレームの表示
                if not self.processed_frame_queue.empty():
                    processed_frame = self.processed_frame_queue.get()
                    cv2.imshow('Processed Frame', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.running = False
            self.camera.release()
            cv2.destroyAllWindows()
            self.client_socket.close()
            receive_thread.join()

if __name__ == '__main__':
    client = CameraClient()
    client.run()