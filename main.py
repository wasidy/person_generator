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

from PIL import Image, ImageFilter
from pillow_lut import load_cube_file
import numpy as np
from utils.face_detector import FaceDetector
from utils.pipelines import ClipSegmentation
from utils.imageutils import image_dilate
import cv2
from datetime import datetime
import re
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer


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


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False


def smooth_mask(image, smooth_size=10, expand=10):
    if expand > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand, expand))
        image = cv2.dilate(image, kernel, iterations=1)
    if smooth_size > 0:
        image = cv2.blur(image, (smooth_size, smooth_size))
    return image


def load_trigger_words(lora_file_name):
    ''' Looking for lora_file_name.txt and reading trigger words '''

    fname = os.path.splitext(lora_file_name)[0]+'.txt'
    if os.path.exists(fname):
        try:
            f = open(fname)
        except Warning:
            print('Trigger words not found')
            words = ''
        else:
            with f:
                words = f.readline()
    else:
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
        self.lora_face_path = config['lora_face_folder']
        self.active_pipe = 'text2img'

    def load_checkpoint(self, filename):
        try:
            self.pipe = StableDiffusionPipeline.from_single_file(
                 self.path+'/'+filename,
                 torch_dtype=torch.float16,
                 use_safetensors=True,
                 safety_checker=None
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
            self.pipe.safety_checker = disabled_safety_checker
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
        if lora_name !='None':
            lora_name = 'face/' + lora_name
        # lora_name = self.lora_person_path + lora_name

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

            # matching colors
            cm = ColorMatcher()
            img_matched_colors = cm.transfer(src=np.array(generated_image, dtype=np.uint8),
                                             ref=faces[0].image,
                                             method='mkl')
            img_matched_colors = Normalizer(img_matched_colors).uint8_norm()

            mask = image_dilate(mask, 10)
            
            #mask = cv2.blur(mask, (10, 10))
            alpha = Image.fromarray(mask, mode='L')
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=5))
            
            new_image = input_image.copy()
            new_image[y1:y2, x1:x2, :] = np.array(img_matched_colors, dtype=np.uint8)
            # new_image[y1:y2, x1:x2, :] = np.array(generated_image, dtype=np.uint8)
            new_image_wf = Image.fromarray(new_image)
            target = Image.fromarray(input_image, mode='RGB')
            target.paste(new_image_wf, (0, 0), mask=alpha)
        return target, alpha

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
                             safety_checker=None,
                             num_inference_steps=steps,
                             height=(height//8)*8, width=(width//8)*8,
                             cross_attention_kwargs={"scale": lora_scale},
                             g_scale=g_scale, **args).images[0]

        if len(current_pipe.get_active_adapters()) > 0:
            current_pipe.delete_adapters(adapter_name)
        return image, seed

def add_noise(img):
    
    pass

def postprocess(input_img):
    img = Image.fromarray(input_img, mode='RGB')
    # img.save('temp/temp.jpg', format='JPEG', subsampling=0, quality=100)
    img.save('temp/temp.jpg', format='JPEG', quality=60)
    img = Image.open('temp/temp.jpg')
    
    lut = load_cube_file('lut/Fuji F125 Kodak 2395 (by Adobe).cube')
    
    processed_image = img.filter(lut)
    processed_image = processed_image.filter(ImageFilter.UnsharpMask(radius=1, percent=30, threshold=3))
    
    processed_image = np.array(processed_image, dtype=np.uint8)
    
    temp_img1 = processed_image / 2.0
    temp_img2 = input_img / 2.0
    fin_img = (temp_img1 + temp_img2).astype(dtype=np.uint8)
    print('Image postprocessed with LUT')
    return fin_img


def ui(pipe, config):
    checkpoints_path = config['checkpoints_folder']
    lora_path = config['lora_folder']
    lora_face_path = config['lora_face_folder']
    checkpoints = get_file_list(checkpoints_path, ['.safetensors'])
    loras = ['None'] + get_file_list(lora_path, ['.safetensors'])
    loras_face = ['None'] + get_file_list(lora_face_path, ['.safetensors'])
    images_bin = []

    with gr.Blocks() as demo:
        input_null_image = gr.Image(visible=False)
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    checkpoint = gr.Dropdown(label='Checkpoint',
                                             choices=checkpoints,
                                             value=checkpoints[0],
                                             interactive=True)
                    lora = gr.Dropdown(label='Lora',
                                       choices=loras,
                                       value=loras[0],
                                       interactive=True
                                       )
                    lora_weight_base = gr.Slider(label='Weight:',
                                                 value=90,
                                                 minimum=0,
                                                 maximum=100,
                                                 step=1,
                                                 interactive=True)
                with gr.Row():

                    prompt = gr.Textbox(label='Prompt',
                                        value=config['default_prompt'],
                                        lines=2,
                                        )
                    negative_prompt = gr.Textbox(label='Negative prompt',
                                                 value=config['default_negative_prompt'],
                                                 lines=2,
                                                 )
                with gr.Row():
                    generate = gr.Button(value='Generate',
                                         size='lg',
                                         variant='primary',
                                         )
                with gr.Tabs():
                    with gr.TabItem('Img2Img'):
                        with gr.Row():
                            with gr.Column():
                                input_image = gr.Image(label='Imput image for Img2Img')

                            with gr.Column():
                                lora_img2img = gr.Dropdown(label='lora for img2img',
                                                           choices=loras,
                                                           value=loras[0],
                                                           interactive=True
                                                           )
                                lora_weight_img2img = gr.Slider(label='Weight:',
                                                                value=90,
                                                                minimum=0,
                                                                maximum=100,
                                                                step=1,
                                                                interactive=True)
                                denoise_img_strength = gr.Slider(label='Denoise strength',
                                                                 value=0.5,
                                                                 minimum=0,
                                                                 maximum=1,
                                                                 step=0.01,
                                                                 interactive=True)

                                copy_out_to_in = gr.Button(value='Copy OUTPUT to INPUT')
                                generate_img2img = gr.Button(value='GENERATE IMG2IMG',
                                                             variant='primary')
                    with gr.TabItem('Regen face'):

                        with gr.Row():
                            with gr.Column():

                                input_image_face = gr.Image(label='Image for face regen')
                            with gr.Column():
                                lora_face = gr.Dropdown(label='Face lora',
                                                        choices=loras_face,
                                                        value=loras_face[0],
                                                        interactive=True
                                                        )
                                lora_face_weight = gr.Slider(label='Weight:',
                                                             value=90,
                                                             minimum=0,
                                                             maximum=100,
                                                             step=1)
                                denoise_face_strength = gr.Slider(label='Denoise strength',
                                                                  value=0.5,
                                                                  minimum=0,
                                                                  maximum=1,
                                                                  step=0.01)
                                copy_out_to_in_face = gr.Button(value='Copy OUTPUT to INPUT')

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
                with gr.Tabs():
                    with gr.TabItem('Output'):
                        output = gr.Image(label='Output image', sources=None)
                        out_dir = gr.State(config['output_folder'])
                        post_process_button = gr.Button('Postprocess image')
                        save_button = gr.Button('Save image')
                        add_to_bin_button = gr.Button('Add to bin')
        
                        images_bin_gr = gr.Gallery(label='Images bin',
                                                   value=images_bin,
                                                   columns=4,
                                                   allow_preview=False,
                                                   object_fit='contain',
                                                   type='numpy')
                        images_bin_idx = gr.State(value=None)
                        clear_bin_button = gr.Button('Clear bin')
                        delete_image_from_bin_button = gr.Button('Delete image from bin')
                    with gr.TabItem('Postprocess'):
                        postprocessed = gr.Image()
                        save_button_postprocessed = gr.Button('Save image')

        post_process_button.click(fn=postprocess, inputs=[output], outputs=[postprocessed])

        def add_to_bin_button_fn(img):
            images_bin.append(img)
            return images_bin

        add_to_bin_button.click(fn=add_to_bin_button_fn,
                                inputs=[output],
                                outputs=[images_bin_gr])

        def images_bin_select(evt: gr.SelectData):
            bin_idx = evt.index
            img = images_bin[bin_idx]
            print(f'Image {bin_idx} selected')
            return img, bin_idx

        images_bin_gr.select(fn=images_bin_select, outputs=[output, images_bin_idx])

        def clear_bin_fn(idx):
            images_bin = []
            return images_bin, None

        clear_bin_button.click(fn=clear_bin_fn,
                               inputs=[images_bin_idx],
                               outputs=[images_bin_gr, images_bin_idx])

        def delete_image_from_bin(idx):
            if idx is not None:
                print(f'Deleting from gallery id {idx}')
                del images_bin[idx]
            return images_bin

        delete_image_from_bin_button.click(fn=delete_image_from_bin,
                                           inputs=[images_bin_idx],
                                           outputs=[images_bin_gr],
                                           )

        save_button.click(fn=save_image, inputs=[output, out_dir])
        save_button_postprocessed.click(fn=save_image, inputs=[postprocessed, out_dir])
        
        copy_out_to_in.click(fn=lambda x: x, inputs=[output], outputs=[input_image])
        copy_out_to_in_face.click(fn=lambda x: x, inputs=[output], outputs=[input_image_face])

        checkpoint.input(pipe.load_checkpoint,
                         inputs=[checkpoint],
                         outputs=[checkpoint],
                         show_progress='full',
                         queue=True)

        generate.click(fn=pipe.generate_image,
                       inputs=[prompt,
                               negative_prompt,
                               input_null_image,
                               denoise_img_strength,
                               width,
                               height,
                               g_scale,
                               seed,
                               steps,
                               lora,
                               lora_weight_base,
                               gr_scheduler],
                       outputs=[output, last_seed])

        generate_img2img.click(fn=pipe.generate_image,
                               inputs=[prompt,
                                       negative_prompt,
                                       input_image,
                                       denoise_img_strength,
                                       width,
                                       height,
                                       g_scale,
                                       seed,
                                       steps,
                                       lora_img2img,
                                       lora_weight_img2img,
                                       gr_scheduler],
                               outputs=[output, last_seed])

        regen_face.click(fn=pipe.regen_face,
                         inputs=[prompt,
                                 negative_prompt,
                                 input_image_face,
                                 output,
                                 denoise_face_strength,
                                 width,
                                 height,
                                 g_scale,
                                 seed,
                                 steps,
                                 lora_face,
                                 lora_face_weight,
                                 gr_scheduler],
                         outputs=[output, postprocessed])

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
    demo.launch(server_name='0.0.0.0')
