# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:32:01 2024

@author: Vasiliy Stepanov
"""

# OpenCLIP ViT-bigG/14 SDXL text encoder
# CLIP ViT-L/14 SD1.5 text encoder

from clip_interrogator import Config, Interrogator
import gradio as gr
import numpy as np
from PIL import Image

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

fname = 'prompts.txt'

def interrogate(image):
    image = Image.fromarray(image)
    prompt = ci.interrogate(image)
    with open(fname, 'a', encoding="utf-8") as f:
        f.write(prompt+'\n')
        
    return prompt

with gr.Blocks() as demo:
    input_image = gr.Image()
    ci_button = gr.Button('Interrogate!')
    output_text = gr.Text()
    
    ci_button.click(fn=interrogate, inputs=[input_image], outputs=[output_text])

if __name__ == '__main__':
    demo.launch()

