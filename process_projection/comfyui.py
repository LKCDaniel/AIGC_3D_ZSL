import os
import uuid

import websocket
import json
import urllib.request
import urllib.parse
import mimetypes
import io
from PIL import Image
from tqdm import tqdm


class Comfyui:

    def __init__(self, server_address="127.0.0.1:8188", workflow="controlnet"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.connected = False
        self.workflow_path = os.path.join('process_projection', f'{workflow}.json')
        self.resolution = 512
        self.connect()

    def set_resolution(self, resolution):
        self.resolution = resolution

    def connect(self):
        if not self.connected:
            self.ws = websocket.WebSocket()
            self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
            self.connected = True
            print("connected to ComfyUI WebSocket")
        return self.connected

    def disconnect(self):
        if self.connected and self.ws:
            self.ws.close()
            self.connected = False
            print("closed connection to ComfyUI WebSocket")

    def _upload_image(self, image_path):
        filename = os.path.basename(image_path)
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        with open(image_path, 'rb') as f:
            image_data = f.read()

        boundary = '----WebKitFormBoundary' + str(uuid.uuid4())
        data = [f'--{boundary}\r\n'.encode('utf-8'),
                f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode('utf-8'),
                f'Content-Type: {mime_type}\r\n\r\n'.encode('utf-8'), image_data,
                f'\r\n--{boundary}--\r\n'.encode('utf-8')]
        body = b''.join(data)

        headers = {
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(body))
        }

        req = urllib.request.Request(
            f"http://{self.server_address}/upload/image",
            data=body,
            headers=headers,
            method='POST'
        )

        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))

    def _queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def _get_images(self, prompt):
        if not self.connected:
            self.connect()

        prompt_id = self._queue_prompt(prompt)['prompt_id']
        output_images = {}
        current_node = ""
        pbar = None

        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['prompt_id'] == prompt_id:
                        if data['node'] is None:
                            if pbar:
                                pbar.close()
                            break
                        else:
                            current_node = data['node']
                elif message['type'] == 'progress':
                    current_value = message['data']['value']
                    max_value = message['data']['max']
                    if pbar is None:
                        pbar = tqdm(total=max_value, desc="SD processing", unit="step")
                        pbar.update(current_value)
                    else:
                        pbar.n = current_value
                        pbar.refresh()
            else:
                if current_node == 'save_image_websocket_node':
                    images_output = output_images.get(current_node, [])
                    images_output.append(out[8:])
                    output_images[current_node] = images_output

        return output_images

    def _create_workflow(self, input_image_filename, prompt, style):
        with open(self.workflow_path, 'r') as f:
            workflow = json.load(f)
        workflow["input"]["inputs"]["image"] = input_image_filename
        workflow["prompts"]["inputs"]["text"] = prompt
        workflow["rescale"]["inputs"]["width"] = self.resolution
        workflow["rescale"]["inputs"]["height"] = self.resolution
        workflow["latentSpace"]["inputs"]["width"] = self.resolution
        workflow["latentSpace"]["inputs"]["height"] = self.resolution
        match style:
            case "depth":
                controlnet = 'control_v11f1p_sd15_depth.pth'
            case 'edge':
                controlnet = 'control_v11p_sd15_canny.pth' # scribble, canny
            case 'normal':
                controlnet = 'control_v11p_sd15_normalbae.pth'
            case _: # render
                controlnet = ''

        workflow["ControlNetLoader"]["inputs"]["control_net_name"] = controlnet
        return workflow

    def process_image(self, prompt, style, input_path, output_path):
        upload_result = self._upload_image(input_path)
        input_image_name = upload_result['name']

        workflow = self._create_workflow(input_image_name, prompt, style)

        results = []
        images = self._get_images(workflow)

        for node_id in images:
            for image_data in images[node_id]:
                image = Image.open(io.BytesIO(image_data))
                image.save(output_path)
                results.append(output_path)
                print(f"Processed {os.path.basename(input_path)}, prompt: {prompt}, saved to {output_path}...")
                break

        return results



if __name__ == "__main__":

    if 1:
        comfyui = Comfyui()
        try:
            # 连接到ComfyUI
            comfyui.connect()

            # 处理图像，每个类别生成3张图片
            class_name = "ant"  # 替换为目标类别
            workflow_path = "comfyui_workflows"  # 替换为工作流路径
            comfyui_output_dir = "comfyui_processed"  # 替换为ComfyUI输出目录

            processed_images = comfyui.process_image(
                output_image_path,
                class_name,
                comfyui_output_dir,
                run_count=3  # 生成3张图片
            )

            print(f"生成了 {len(processed_images)} 张处理后的图像")

        finally:
            # 断开连接
            comfyui.disconnect()
