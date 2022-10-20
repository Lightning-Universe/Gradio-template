import lightning as L
from lightning.app.components.serve import ServeGradio
import gradio as gr

class LitGradio(ServeGradio):

    inputs = gr.inputs.Textbox(default='lightning', label='name input')
    outputs = gr.outputs.Textbox(label='output')
    examples = [["hello lightning"]]

    def predict(self, input_text):
        return self.model(input_text)

    def build_model(self):
        fake_model = lambda x: f"hello {x}"
        return fake_model

class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_gradio = LitGradio()

    def run(self):
        self.lit_gradio.run()

    def configure_layout(self):
        return [{"name": "home", "content": self.lit_gradio}]

app = L.LightningApp(RootFlow())
