import gradio as gr
import os
import torch
import json

from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler, KDPM2DiscreteScheduler, DPMSolverSinglestepScheduler
from diffusers import DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler
from diffusers import DDPMScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler
from diffusers import PNDMScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler

from PIL import Image
import numpy as np
from utils.face_detector import FaceDetector
from utils.pipelines import ClipSegmentation
from utils.imageutils import image_dilate
import cv2
from datetime import datetime
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    print('GPU not found, using CPU!')

schedulers = {
    'DDIMScheduler': DDIMScheduler,
    'KDPM2DiscreteScheduler': KDPM2DiscreteScheduler,
    'DPMSolverSinglestepScheduler': DPMSolverSinglestepScheduler,
    'DEISMultistepScheduler': DEISMultistepScheduler,
    'KDPM2AncestralDiscreteScheduler': KDPM2AncestralDiscreteScheduler,
    'DDPMScheduler': DDPMScheduler,
    'EulerDiscreteScheduler': EulerDiscreteScheduler,
    'HeunDiscreteScheduler': HeunDiscreteScheduler,
    'PNDMScheduler': PNDMScheduler,
    'LMSDiscreteScheduler': LMSDiscreteScheduler,
    'UniPCMultistepScheduler': UniPCMultistepScheduler,
    'EulerAncestralDiscreteScheduler': EulerAncestralDiscreteScheduler,
    'DPMSolverMultistepScheduler': DPMSolverMultistepScheduler
}


def smooth_mask(image, smooth_size=10, expand=10):
    if expand > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand, expand))
        image = cv2.dilate(image, kernel, iterations=1)
    if smooth_size > 0:
        image = cv2.blur(image, (smooth_size, smooth_size))
    return image


def load_trigger_words(lora_file_name):
    ''' Looking for lora_file_name.txt and reading trigger words '''
    try:
        with open(os.path.splitext(lora_file_name)[0]+'.txt') as f:
            words = f.readline()
    except Warning:
        print('Trigger words not found')
        words = ''
    return words


def get_file_list(path, ext):
    flist = os.listdir(path)
    files = [file for file in flist if os.path.splitext(file)[1] in ext]
    if len(files) == 0:
        raise FileNotFoundError()
        print('Put Stable Diffusion cpkt or safetensors checkpoints to folder!')
    return files


def save_image(image, path):
    img = Image.fromarray(image, mode='RGB')
    c_time = re.sub('[:. -]', '', str(datetime.today()))
    f_name = path + c_time + '.png'
    try:
        img.save(f_name)
        print(f'{f_name} saved!')
    except Warning:
        print(f'Error writing {f_name}')
        return None
    return None


class SDPipe():
    def __init__(self, config):
        self.path = config['checkpoints_folder']
        self.checkpoints = get_file_list(self.path, ['.safetensors'])
        self.load_checkpoint(self.checkpoints[0])
        self.lora_path = config['lora_folder']
        self.active_pipe = 'text2img'

    def load_checkpoint(self, filename):
        try:
            self.pipe = StableDiffusionPipeline.from_single_file(
                 self.path+'/'+filename,
                 torch_dtype=torch.float16,
                 use_safetensors=True
                 )
            self.img2img = StableDiffusionImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False)

            self.pipe = self.pipe.to(device)
            self.img2img = self.img2img.to(device)

            print(f'Checkpoint {self.path + filename} loaded!')
        except Warning:
            print('Checkpoint corrupted!')
        return filename

    def load_lora(self, lora_name):
        self.pipe.load_lora_weights(self.lora_path + lora_name,
                                    weight_name=self.lora_path + lora_name,
                                    adapter_name='custom_lora')

    def regen_face(self, positive_prompt, negative_prompt, input_image, output_image,
                   denoise_strength, width,
                   height, g_scale, manual_seed, steps, lora_name, lora_weight,
                   scheduler_name):
        if not isinstance(input_image, np.ndarray):
            if not isinstance(output_image, np.ndarray):
                print('No image')
                return None
            else:
                input_image = output_image

        mask, faces = face_detector(input_image, expand_value=1.4)

        if len(faces) > 0:
            x1, y1, x2, y2 = faces[0].coords
            gen_size = 512
            face_size = faces[0].image.shape[0]
            gen_img, seed = self.generate_image(positive_prompt,
                                                negative_prompt,
                                                faces[0].image,
                                                denoise_strength,
                                                gen_size,
                                                gen_size,
                                                g_scale,
                                                manual_seed,
                                                steps,
                                                lora_name,
                                                lora_weight,
                                                scheduler_name
                                                )
            generated_image = gen_img.resize((face_size, face_size), Image.BICUBIC)
            mask = image_dilate(mask, 20)
            mask = cv2.blur(mask, (20, 20))
            alpha = Image.fromarray(mask, mode='L')
            new_image = input_image.copy()
            new_image[y1:y2, x1:x2, :] = np.array(generated_image, dtype=np.uint8)
            new_image_wf = Image.fromarray(new_image)
            target = Image.fromarray(input_image, mode='RGB')
            target.paste(new_image_wf, (0, 0), mask=alpha)
        return target

    def generate_image(self, positive_prompt, negative_prompt, input_image,
                       denoise_strength, width,
                       height, g_scale, manual_seed, steps, lora_name, lora_weight,
                       scheduler_name):
        generator = torch.Generator(device=device)
        seed = generator.seed() if manual_seed == -1 else manual_seed
        generator = generator.manual_seed(seed)
        args = dict()

        if isinstance(input_image, np.ndarray):
            print('Img2Img mode')
            current_pipe = self.img2img
            image = Image.fromarray(input_image, mode='RGB')
            image = image.resize((width, height), Image.LANCZOS)
            args = {'image': image, 'strength': denoise_strength}

        else:
            print('Text2Img mode')
            current_pipe = self.pipe

        if isinstance(lora_name, str):
            if lora_name != 'None':
                print(f'Lora loaded: {lora_name}')
                adapter_name = lora_name[0:lora_name.find('.')]
                current_pipe.load_lora_weights(self.lora_path + lora_name,
                                               weight_name=self.lora_path + lora_name,
                                               adapter_name=adapter_name)
                trigger_words = load_trigger_words(self.lora_path + lora_name)
                positive_prompt = positive_prompt + ', ' + trigger_words
                print(f'New prompt: {positive_prompt}')
                current_pipe.set_adapters(adapter_name)

        print(f'Active adapters: {self.pipe.get_active_adapters()}')
        lora_scale = lora_weight/100

        current_pipe.scheduler = schedulers[scheduler_name].from_config(self.pipe.scheduler.config)
        image = current_pipe(prompt=positive_prompt,
                             negative_prompt=negative_prompt,
                             generator=generator,
                             num_inference_steps=steps,
                             height=(height//8)*8, width=(width//8)*8,
                             cross_attention_kwargs={"scale": lora_scale},
                             g_scale=g_scale, **args).images[0]

        if len(current_pipe.get_active_adapters()) > 0:
            current_pipe.delete_adapters(adapter_name)
        return image, seed


def ui(pipe, config):
    checkpoints_path = config['checkpoints_folder']
    lora_path = config['lora_folder']
    checkpoints = get_file_list(checkpoints_path, ['.safetensors'])
    loras = ['None'] + get_file_list(lora_path, ['.safetensors'])

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    checkpoint = gr.Dropdown(label='Checkpoint',
                                             choices=checkpoints,
                                             value=checkpoints[0],
                                             interactive=True)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        prompt = gr.Textbox(label='Prompt',
                                            value=config['default_prompt'],
                                            lines=2,
                                            )
                        negative_prompt = gr.Textbox(label='Negative prompt',
                                                     value=config['default_negative_prompt'],
                                                     lines=2,
                                                     )
                    with gr.Column(scale=1, min_width=100):
                        generate = gr.Button(value='Generate',
                                             size='lg',
                                             variant='primary')
                with gr.Group():
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=3):
                            lora = gr.Dropdown(label='Lora',
                                               choices=loras,
                                               value=loras[0],
                                               interactive=True
                                               )
                        with gr.Column(scale=1, min_width=100):
                            lora_weight = gr.Slider(label='Weight:',
                                                    value=90,
                                                    minimum=0,
                                                    maximum=100,
                                                    step=1)
                with gr.Accordion('Img2Img'):
                    with gr.Row():
                        with gr.Column():
                            input_null_image = gr.Image(visible=False)
                            input_image = gr.Image(label='Imput image for Img2Img')
                        with gr.Column():
                            denoise_strength = gr.Slider(label='Denoise strength',
                                                         value=0.7,
                                                         minimum=0,
                                                         maximum=1,
                                                         step=0.01)
                            copy_out_to_in = gr.Button(value='Copy OUTPUT to INPUT')
                            generate_img2img = gr.Button(value='GENERATE IMAGE',
                                                         variant='primary')
                            regen_face = gr.Button(value='GENERATE FACE',
                                                   variant='primary')
    
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion('Generation settings'):
                            schedulers_list = list(schedulers.keys())
                            gr_scheduler = gr.Dropdown(label='Scheduler',
                                                       choices=schedulers_list,
                                                       value=schedulers_list[0])
                            g_scale = gr.Slider(label='Guidance scale',
                                                value=7.5,
                                                minimum=1,
                                                maximum=20)
    
                            steps = gr.Slider(label='Steps',
                                              value=30,
                                              minimum=1,
                                              maximum=100,
                                              step=1
                                              )
                            seed = gr.Number(label='Seed',
                                             value=-1,
                                             precision=0,
                                             interactive=True,
                                             minimum=-1)
                            last_seed = gr.State(value=-1)
                            reuse_last_seed = gr.Button(value="Reuse last seed")
                            reset_seed = gr.Button(value='Reset seed')
                    with gr.Column():
                        with gr.Group():
                            width = gr.Slider(label='Width',
                                              value=512,
                                              minimum=256,
                                              maximum=2048,
                                              step=8)
                            height = gr.Slider(label='Height',
                                               value=768,
                                               minimum=256,
                                               maximum=2048,
                                               step=8)
            with gr.Column():
                output = gr.Image(label='Output image')
                out_dir = gr.State(config['output_folder'])
                save_button = gr.Button('Save image')

        save_button.click(fn=save_image, inputs=[output, out_dir])
    
        copy_out_to_in.click(fn=lambda x: x, inputs=[output], outputs=[input_image])
    
        checkpoint.input(pipe.load_checkpoint,
                         inputs=[checkpoint],
                         outputs=[checkpoint],
                         show_progress='full',
                         queue=True)
    
        generate.click(fn=pipe.generate_image,
                       inputs=[prompt,
                               negative_prompt,
                               input_null_image,
                               denoise_strength,
                               width,
                               height,
                               g_scale,
                               seed,
                               steps,
                               lora,
                               lora_weight,
                               gr_scheduler],
                       outputs=[output, last_seed])
    
        generate_img2img.click(fn=pipe.generate_image,
                               inputs=[prompt,
                                       negative_prompt,
                                       input_image,
                                       denoise_strength,
                                       width,
                                       height,
                                       g_scale,
                                       seed,
                                       steps,
                                       lora,
                                       lora_weight,
                                       gr_scheduler],
                               outputs=[output, last_seed])
    
        regen_face.click(fn=pipe.regen_face,
                         inputs=[prompt,
                                 negative_prompt,
                                 input_image,
                                 output,
                                 denoise_strength,
                                 width,
                                 height,
                                 g_scale,
                                 seed,
                                 steps,
                                 lora,
                                 lora_weight,
                                 gr_scheduler],
                         outputs=[output])
    
        reset_seed.click(fn=lambda x: -1, outputs=[seed])
        reuse_last_seed.click(fn=lambda x: x, inputs=[last_seed], outputs=[seed])

    return demo


if __name__ == '__main__':
    # Load config
    try:
        f = open('config.json', 'r')
        config = json.load(f)
    except FileNotFoundError:
        print('Config file not found')
    
    clip_predict = ClipSegmentation('CIDAS/clipseg-rd64-refined', 'CIDAS/clipseg-rd64-refined')
    face_detector = FaceDetector(clip_predict)
    pipe = SDPipe(config)
    
    demo = ui(pipe, config)
    demo.launch()
