# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:36:55 2024

@author: Vasiliy Stepanov
"""

import random
import gradio as gr

names = []
surnames = []

with open('data/names.txt') as fnames:
    names = [line.rstrip('\n') for line in fnames]

with open('data/surnames.txt') as fsurnames:
    surnames = [line.rstrip('\n') for line in fsurnames]

names_per_prompt = 4
sur_names_per_prompt = 4

'''
Prompt generating:
    1. Combination of names and surnames
    2. Age
    3. Hair color
    4. Shape of face
    5. Eyes
    6. Mouth
    7. Nose
    8. Skin tone
    9. Randomize prompt with light and front/side view
    10. Negative prompt with hands, bad skin and etc
    11. Freckles to positive or negative
    12. Makeup
'''

gender = ['woman', 'man']
gender_prompt=['1girl', '1man']
face_shapes = ['oval', 'square', 'round', 'heart', 'rectangular', 'diamond']
eye_colors = ['agate', 'baby blue', 'lapis', 'nomad', 'nordic', 'liz', 'frisco', 'cobalt', 'azure', 'vale', 'hazel', 'ebony']
mouth = ['full lips', 'heavy upper lips', 'wide lips', 'round lips', 'heavy lower lips', 'thin lips', 'bow-shaped lips', 'heart-shaped lips', 'glamour', 'beestung']
nose = ['roman', 'greek', 'snub', 'turned-up', 'straight', 'flat', 'short', 'turned-up']
skin_tone = ['fair', 'light', 'medium', 'dark']
hair_color = ['blond', 'brunette', 'brown', 'black', 'red', 'color shade', 'auburn']
hair_length = ['short', 'bob', 'medium', 'long']
hair_style = ['pixie cut', 'feathered', 'french twist', 'bob', 'tousled', 'braid', 'curved', 'bangs', 'straight']

additional_prompt = 'photography, detailed skin, high quality'

with gr.Blocks() as demo:
    with gr.Row():
        final_prompt_gr = gr.Text(value='', label='Prompt', show_label=True)
    with gr.Row():
        names_nums_gr = gr.Slider(minimum=1,
                                  maximum=8,
                                  value=4,
                                  step=1,
                                  label='Names',
                                  interactive=True
                                  )
        surnames_nums_gr = gr.Slider(minimum=1,
                                     maximum=8,
                                     value=4,
                                     step=1,
                                     label='Surnames',
                                     interactive=True
                                     )
    with gr.Row():
        gender_gr = gr.Radio(choices=gender, value=gender[0], label='Gender')
        face_shape_gr = gr.Radio(choices=face_shapes, label='Face shape')
        eye_color_gr = gr.Radio(choices=eye_colors, label='Eye color')
        mouth_gr = gr.Radio(choices=mouth, label='Mouth')
        nose_gr = gr.Radio(choices=nose, label='Nose')
        skin_tone_gr = gr.Radio(choices=skin_tone, label='Skin tone')
        hair_color_gr = gr.Radio(choices=hair_color, label='Hair color')
        hair_length_gr = gr.Radio(choices=hair_length, label='Hair length')
        hair_style_gr = gr.Radio(choices=hair_style, label='Hair style')
        additional_prompt_gr = gr.Text(value=additional_prompt, label='Additional prompt')
        
    with gr.Row():
        generate_prompt_gr = gr.Button('Generate prompt')
    
    def generate_prompt(names_nums,
                        surnames_nums,
                        gender,
                        face_shape,
                        eye_color,
                        mouth,
                        nose,
                        skin_tone,
                        hair_color,
                        hair_length,
                        hair_style,
                        additional_prompt):
        random_names = []
        random_surnames = []
        for _ in range(names_nums):
            random_names.append(random.choice(names))
        
        for _ in range(surnames_nums):
            random_surnames.append(random.choice(surnames))
        
        full_names = random_names + random_surnames
        random.shuffle(full_names)
        full_names = ', '.join(full_names)
        
        prompt = f'A portrait of {gender}, {full_names}, with {eye_color} eyes, {mouth} mouth, {nose} nose, {face_shape} face, {skin_tone} skin, {hair_length} {hair_color} {hair_style} hair, {additional_prompt}'
        
        
        return prompt
    
    generate_prompt_gr.click(fn=generate_prompt,
                             inputs=[
                                 names_nums_gr,
                                 surnames_nums_gr,
                                 gender_gr,
                                 face_shape_gr,
                                 eye_color_gr,
                                 mouth_gr,
                                 nose_gr,
                                 skin_tone_gr,
                                 hair_color_gr,
                                 hair_length_gr,
                                 hair_style_gr,
                                 additional_prompt_gr
                                 ],
                             outputs=[final_prompt_gr]
                             )
    
if __name__ == '__main__':
    #print(f'{random.choice(names)}, {random.choice(surnames)}, ')
    demo.launch()
    