# server.py
import socket
import numpy as np
import cv2
import torch
import threading
import msgpack
import msgpack_numpy as m
m.patch()

class ImageProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernel = np.ones((5,5), np.uint8)
        # 前回のフレームをキャッシュ
        self.prev_frame = None
        print(f"Using device: {self.device}")

    def process_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        binary_tensor = torch.from_numpy(binary).float().to(self.device)
        kernel_tensor = torch.from_numpy(self.kernel).float().to(self.device)

        binary_tensor = binary_tensor.unsqueeze(0).unsqueeze(0)
        kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)

        padding = (kernel_tensor.shape[2]//2, kernel_tensor.shape[3]//2)
        dilated = torch.nn.functional.conv2d(
            binary_tensor,
            kernel_tensor,
            padding=padding
        )

        dilated = (dilated > 0).float()
        dilated = dilated.squeeze().cpu().numpy().astype(np.uint8) * 255
        dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

        return dilated_rgb

class StreamHandler(threading.Thread):
    def __init__(self, client_socket, processor):
        super().__init__()
        self.client_socket = client_socket
        self.processor = processor
        self.running = True

    def run(self):
        try:
            buffer = bytearray()
            while self.running:
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

                frame_data = msgpack.unpackb(msg_data)
                frame = np.asarray(frame_data)

                # 画像処理
                processed = self.processor.process_image(frame)

                # 処理結果の圧縮と送信
                processed_data = msgpack.packb(processed)
                size_bytes = len(processed_data).to_bytes(4, 'big')

                self.client_socket.sendall(size_bytes + processed_data)

        except Exception as e:
            print(f"Error in stream handler: {e}")
        finally:
            self.client_socket.close()

def start_server(host='0.0.0.0', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)

    processor = ImageProcessor()
    print(f"Server started on {host}:{port}")

    try:
        while True:
            client_socket, addr = server_socket.accept()
            print(f"New connection from {addr}")
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            handler = StreamHandler(client_socket, processor)
            handler.start()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        server_socket.close()

if __name__ == '__main__':
    start_server()