from functools import partial
import gradio as gr
import requests
import torch
from PIL import Image

import lightning as L
from lightning.app.components.serve import ServeGradio


class LitGradio(ServeGradio):

    inputs = gr.inputs.Image(type="pil")
    outputs = gr.outputs.Image(type="pil")
    demo_img = "https://cdn.gobankingrates.com/wp-content/uploads/2022/02/shutterstock_editorial_10298578j-1.jpg"
    img = Image.open(requests.get(demo_img, stream=True).raw)
    img.save("shutterstock_editorial_10298578j-1.jpg")
    examples = [["shutterstock_editorial_10298578j-1.jpg"]]

    def __init__(self):
        super().__init__()
        self.ready = False

    def predict(self, img):
        return self.model(img=img)

    def build_model(self):
        repo = "AK391/animegan2-pytorch:main"
        model = torch.hub.load(repo, "generator", device="cpu")
        face2paint = torch.hub.load(repo, "face2paint", size=512, device="cpu")
        self.ready = True
        return partial(face2paint, model=model)


app = L.LightningApp(LitGradio())
