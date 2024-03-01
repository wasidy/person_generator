import gradio as gr
import os
import torch
import json
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import DDIMScheduler, KDPM2DiscreteScheduler, DPMSolverSinglestepScheduler
from diffusers import DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler
from diffusers import DDPMScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler
from diffusers import PNDMScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.loaders import FromSingleFileMixin

from PIL import Image
import numpy as np
from utils.face_detector import FaceDetector
from utils.pipelines import ClipSegmentation
from utils.imageutils import image_blur, image_dilate
import cv2


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device=='cpu':
    print('GPU not found, using CPU!')

try:
    f = open('config.json', 'r')
    config = json.load(f)
except FileNotFoundError:
    print('File not found')

default_prompt = "sexy young woman, sitting on beach"
default_negative_prompt = "bad quality, ugly, wrong anatomy"
test_image = Image.open('D:/Images/test.png')

schedulers_names = [
    'DDIMScheduler',
    'KDPM2DiscreteScheduler',
    'DPMSolverSinglestepScheduler',
    'DEISMultistepScheduler',
    'KDPM2AncestralDiscreteScheduler',
    'DDPMScheduler',
    'EulerDiscreteScheduler',
    'HeunDiscreteScheduler',
    'PNDMScheduler',
    'LMSDiscreteScheduler',
    'UniPCMultistepScheduler',
    'EulerAncestralDiscreteScheduler',
    'DPMSolverMultistepScheduler'
    ]

schedulers = {
    'DDIMScheduler' : DDIMScheduler,
    'KDPM2DiscreteScheduler' : KDPM2DiscreteScheduler,
    'DPMSolverSinglestepScheduler' : DPMSolverSinglestepScheduler,
    'DEISMultistepScheduler' : DEISMultistepScheduler,
    'KDPM2AncestralDiscreteScheduler' : KDPM2AncestralDiscreteScheduler,
    'DDPMScheduler' : DDPMScheduler,
    'EulerDiscreteScheduler' : EulerDiscreteScheduler,
    'HeunDiscreteScheduler' : HeunDiscreteScheduler,
    'PNDMScheduler' : PNDMScheduler,
    'LMSDiscreteScheduler' : LMSDiscreteScheduler,
    'UniPCMultistepScheduler' : UniPCMultistepScheduler,
    'EulerAncestralDiscreteScheduler' : EulerAncestralDiscreteScheduler,
    'DPMSolverMultistepScheduler' : DPMSolverMultistepScheduler
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
    except:
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


class SDPipe():
    def __init__(self, config):
        self.path = config['checkpoints_folder']
        self.checkpoints = get_file_list(self.path, ['.safetensors'])
        self.load_checkpoint(self.checkpoints[0])
        self.lora_path = config['lora_folder']
        self.active_pipe = 'text2img'

    def load_checkpoint(self, filename):
        try:
            # self.pipe = DiffusionPipeline.from_single_file(
            #     self.path+'/'+filename,
            #     torch_dtype=torch.float16,
            #     use_safetensors=True
            #     )

            # models = {k: v for k,v in vars(pipeline).items() if not k.startswith("_")}

            self.pipe = StableDiffusionPipeline.from_single_file(
                 self.path+'/'+filename,
                 torch_dtype=torch.float16,
                 use_safetensors=True
                 )
            #sub_models = {k: v for k,v in vars(self.pipe).items() if not k.startswith("_")}
            #print(sub_models)
            #self.img2img = StableDiffusionImg2ImgPipeline(unet=self.pipe.unet,
#                                                          vae=self.pipe.vae,
#                                                          )

            #self.text2img = StableDiffusionPipeline(**models)
            self.img2img = StableDiffusionImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False)
            #self.inpaint = StableDiffusionInpaintPipeline(**models)

            #self.pipe = StableDiffusionImg2ImgPipeline.from_single_file(self.path+'/'+filename,
            #                                                     #torch_dtype=torch.float16,
            #                                                     use_safetensors=True)

            self.pipe = self.pipe.to(device)
            self.img2img = self.img2img.to(device)
            #self.text2img = self.text2img.to(device)
            #self.img2img = self.img2img.to(device)
            #self.inpaint = self.inpaint.to(device)
            print(f'Checkpoint {self.path + filename} loaded!')
        except Warning:
            print('Checkpoint corrupted!')

        return filename

    def load_lora(self, lora_name):
        self.pipe.load_lora_weights(self.lora_path + lora_name,
                                    weight_name=self.lora_path + lora_name,
                                    adapter_name='custom_lora')

    def generate_tile(self, tile, mask, coords):
        
        pass
    
    def regen_face(self, positive_prompt, negative_prompt, input_image,
                   denoise_strength, width,
                   height, g_scale, manual_seed, steps, lora_name, lora_weight,
                   scheduler_name):
        if not isinstance(input_image, np.ndarray):
            return None
        mask, faces = face_detector(input_image, expand_value=1.4)
        if len(faces)>0:
            x1, y1, x2, y2 = faces[0].coords
            
            tile_size = max((x2-x1),(y2-y1))
            center_x = (x2-x1)//2 + x1
            center_y = (y2-y1)//2 + y1
            
            gen_size = 512
            face_size = faces[0].image.shape[0]
            
            #print(f'Original tile size: {tile_size}, face image size: {faces[0].image.shape}')
            gen_img = self.generate_image(positive_prompt,
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
            print(type(gen_img))
            generated_image = gen_img
            #generated_image = Image.fromarray(gen_img, mode='RGB')
            generated_image = generated_image.resize((face_size, face_size), Image.BICUBIC)
            #face_mask = smooth_mask(faces[0].mask, 10, 10)
            #face_mask = faces[0].mask
            mask = image_dilate(mask,20)
            mask = cv2.blur(mask, (20,20))
            alpha = Image.fromarray(mask, mode='L')
            #print(face_mask, face_mask.max(), face_mask.shape)
            
            #generated_image.putalpha(alpha)
            new_image = input_image.copy()
            new_image[y1:y2, x1:x2, :] = np.array(generated_image, dtype=np.uint8)
            new_image_wf = Image.fromarray(new_image)
            
            target = Image.fromarray(input_image, mode='RGB')
            #print(target.size, gen)
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
            args={'image': image, 'strength': denoise_strength}
            # add args
        else:
            print('Text2Img mode')
            current_pipe = self.pipe

        #print(f'List of adapters: {self.pipe.get_list_adapters()}')

        if isinstance(lora_name, str):
            print(f'Lora loaded: {lora_name}')
            adapter_name = os.path.splitext(lora_name)[0]
            current_pipe.load_lora_weights(self.lora_path + lora_name,
                                        weight_name=self.lora_path + lora_name,
                                        adapter_name=adapter_name)

            trigger_words = load_trigger_words(self.lora_path + lora_name)
            positive_prompt = positive_prompt + ', ' + trigger_words
            print(f'New prompt: {positive_prompt}')
            # Вынести генерацию имени адаптера из названия файла лоры в отдельную функцию
            current_pipe.set_adapters(adapter_name)
        print(f'Active adapters: {self.pipe.get_active_adapters()}')
        lora_scale = lora_weight/100

        current_pipe.scheduler = schedulers[scheduler_name].from_config(self.pipe.scheduler.config)

        image = current_pipe(prompt=positive_prompt, negative_prompt=negative_prompt,
                                      generator=generator,
                                      num_inference_steps=steps,
                                      height=(height//8)*8, width=(width//8)*8,
                                      cross_attention_kwargs={"scale": lora_scale},
                                      g_scale=g_scale, **args).images[0]

        if len(current_pipe.get_active_adapters()) > 0:
            current_pipe.delete_adapters(adapter_name)
        return image


checkpoints_path = 'D:/SD/'
lora_path = 'D:/Lora/'



checkpoints = get_file_list(checkpoints_path, ['.safetensors'])
loras = get_file_list(lora_path, ['.safetensors'])

pipe = SDPipe(config)
css='''
#small_bt: {width: 100px; border-radius: 5px}
'''
with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                checkpoint = gr.Dropdown(label='Checkpoint',
                                         choices=checkpoints,
                                         value=checkpoints[0],
                                         interactive=True)
            with gr.Group():
                lora = gr.Dropdown(label='Lora',
                                   choices=loras
                                   )
                lora_weight = gr.Slider(label='Lora weight',
                                        value=90,
                                        minimum=0,
                                        maximum=100,
                                        step=1)
            with gr.Group():
                prompt = gr.Textbox(label='Prompt',
                                    value=default_prompt,
                                    lines=2)
                negative_prompt = gr.Textbox(label='Negative prompt',
                                             value=default_negative_prompt,
                                             lines=2)
            generate = gr.Button(value='Generate')
            regen_face = gr.Button(value='ReGen face with lora & prompt')

            with gr.Row():
                with gr.Column():
                    with gr.Group():
                            gr_scheduler = gr.Dropdown(label='Scheduler',
                                                       choices=schedulers_names,
                                                       value=schedulers_names[0])
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


            seed = gr.Number(label='Seed',
                             value=-1,
                             precision=0,
                             interactive=True,
                             minimum=-1)
            reset_seed = gr.Button(value='Reset seed', elem_id='small_bt'
                                   )

            

            with gr.Group():
                input_image = gr.Image(label='Imput image for Img2Img')
                denoise_strength = gr.Slider(label='Denoise strength',
                                             value=0.7,
                                             minimum=0,
                                             maximum=1,
                                             step=0.01)
                copy_out_to_in = gr.Button(value='Copy OUTPUT to INPUT')

            with gr.Row():
                pass
        with gr.Column():
            output = gr.Image(label='Output image')

    copy_out_to_in.click(inputs=[output], outputs=[input_image])

    checkpoint.input(pipe.load_checkpoint,
                     inputs=[checkpoint],
                     outputs=[checkpoint],
                     show_progress='full',
                     queue=True)

    generate.click(fn=pipe.generate_image,
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
                   outputs=[output])

    regen_face.click(fn=pipe.regen_face,
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
                        outputs=[output])


if __name__ == '__main__':
    clip_predict = ClipSegmentation(
    'CIDAS/clipseg-rd64-refined',
    'CIDAS/clipseg-rd64-refined'
    )

    
    face_detector = FaceDetector(clip_predict)
    demo.launch()