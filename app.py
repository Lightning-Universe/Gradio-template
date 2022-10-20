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


app = L.LightningApp(LitGradio())
