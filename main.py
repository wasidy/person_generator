import gradio as gr
import os
import torch
import json
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler, KDPM2DiscreteScheduler, DPMSolverSinglestepScheduler
from diffusers import DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler
from diffusers import DDPMScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler
from diffusers import PNDMScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler

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


    def load_checkpoint(self, filename):
        try:
            self.pipe = StableDiffusionPipeline.from_single_file(self.path+'/'+filename,
                                                                 #torch_dtype=torch.float16,
                                                                 use_safetensors=True)
            self.pipe = self.pipe.to(device)
            print(f'Checkpoint {self.path + filename} loaded!')
        except Warning:
            print('Checkpoint corrupted!')

        return filename

    def load_lora(self, lora_name):
        self.pipe.load_lora_weights(self.lora_path + lora_name,
                                    weight_name=self.lora_path + lora_name,
                                    adapter_name='custom_lora')

    def generate_image(self, positive_prompt, negative_prompt, width,
                       height, g_scale, manual_seed, steps, lora_name, lora_weight,
                       scheduler_name):
        generator = torch.Generator(device=device)
        seed = generator.seed() if manual_seed == -1 else manual_seed
        generator = generator.manual_seed(seed)
        print(f'List of adapters: {self.pipe.get_list_adapters()}')
        if isinstance(lora_name, str):
            print(f'Lora loaded: {lora_name}')
            adapter_name = os.path.splitext(lora_name)[0]
            
            

            self.pipe.load_lora_weights(self.lora_path + lora_name,
                                        weight_name=self.lora_path + lora_name,
                                        adapter_name=adapter_name)

            trigger_words = load_trigger_words(self.lora_path + lora_name)
            positive_prompt = positive_prompt + ', ' + trigger_words
            print(f'New prompt: {positive_prompt}')
            # Вынести генерацию имени адаптера из названия файла лоры в отдельную функцию
            self.pipe.set_adapters(adapter_name)
        print(f'Active adapters: {self.pipe.get_active_adapters()}')
        lora_scale = lora_weight/100

        self.pipe.scheduler = schedulers[scheduler_name].from_config(self.pipe.scheduler.config)

        image = self.pipe(prompt=positive_prompt, negative_prompt=negative_prompt,
                          generator=generator,
                          num_inference_steps=steps,
                          height=(height//8)*8, width=(width//8)*8,
                          cross_attention_kwargs={"scale": lora_scale},
                          g_scale=g_scale).images[0]
        
        if len(self.pipe.get_active_adapters()) > 0:
            self.pipe.delete_adapters(adapter_name)
        return image

checkpoints_path = 'D:/SD/'
lora_path = 'D:/Lora/'

checkpoints = get_file_list(checkpoints_path, ['.safetensors'])
loras = get_file_list(lora_path, ['.safetensors'])

pipe = SDPipe(config)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem('Text2Img'):
                    prompt = gr.Textbox(label='Prompt',
                                        value=default_prompt,
                                        lines=4)
                    negative_prompt = gr.Textbox(label='Negative prompt',
                                                 value=default_negative_prompt,
                                                 lines=4)
                    checkpoint = gr.Dropdown(label='Checkpoint',
                                             choices=checkpoints,
                                             value=checkpoints[0],
                                             interactive=True)
                    lora = gr.Dropdown(label='Lora',
                                       choices=loras
                                       )
                    lora_weight = gr.Slider(label='Lora weight',
                                            value=90,
                                            minimum=0,
                                            maximum=100,
                                            step=1)
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
                    seed = gr.Number(label='Seed',
                                     value=-1,
                                     precision=0,
                                     interactive=True,
                                     minimum=-1)
                    reset_seed = gr.Button(value='Reset seed')
                    
                    generate = gr.Button(value='Generate')
                with gr.TabItem('Img2Img'):
                    input_image = gr.Image(label='Imput image for Img2Img')



        with gr.Column():
            output = gr.Image(label='Output image')

    checkpoint.input(pipe.load_checkpoint,
                     inputs=[checkpoint],
                     outputs=[checkpoint],
                     show_progress='full',
                     queue=True)

    generate.click(fn=pipe.generate_image,
                   inputs=[prompt,
                           negative_prompt,
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

    demo.launch()