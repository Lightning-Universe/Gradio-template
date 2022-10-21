from functools import partial
import gradio as gr
import requests
import torch
from PIL import Image

import lightning as L
from lightning.app.components.serve import ServeGradio


class LitGradio(ServeGradio):

    inputs = gr.inputs.Image(type="pil", label="Upload Image for Object Detection")
    outputs = gr.outputs.Image(type="pil", label="Output Image")
    demo_img = "https://imageio.forbes.com/specials-images/imageserve/60abf319b47a409ca17f4a3f/Pedestrians-cross-Broadway--in-the-SoHo-neighborhood-in-New-York--United-States--May/960x0.jpg?format=jpg&width=960"
    img = Image.open(requests.get(demo_img, stream=True).raw)
    img.save("960x0.jpg")
    examples = [["960x0.jpg"]]

    def __init__(self):
        super().__init__()
        self.ready = False

    def predict(self, image) -> str:
        results = self.model(image, size=960)  # includes NMS
        results.render()
        return Image.fromarray(results.ims[0])
    
    def build_model(self):
        yolo = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.ready = True
        return yolo

app = L.LightningApp(LitGradio())
