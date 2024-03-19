# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:19:10 2024

@author: Vasiliy Stepanov
"""

import random


# Who?

prefix = [
    'A photo of',]

person = [
    'young woman',
    'young sexy woman',
    'young woman 19 years old',
    ]

# Base appearance

hair = [
    'blond hair',
    'brunette hair',
    'red hair',
    'long blond hair',
    'long brunette curved hair',
    'short brunette hair',
    'bushy brown hair',
    ]

body = [
    'sexy hot body',
    'thin body',
    'big booty',
    'sport body',
    'slim body',
    'fat body',
    'small breast and skinny body',
    'anorexic figure',
    ]

# What doing?

action = [
    'posing',
    'standing',
    'sitting',
    'laying',
    'sitting with legs crossed',
    'showing a leg',
    'smiling',
    ]

# Where?

location =[
    'in bed',
    'in pole',
    'in office',
    'in front of window',
    'in bathroom',
    'on field',
    'on beach',
    'in park',
    'on the side of road',
    'near ocean',
    'under rain',
    'on street',
    'in theater',
    'in museum',
    'in bedroom',
    'in hotel room',
    'in household',
    ]

# How?

clothes = [
    'in white lingerie',
    'in black bra top and jeans',
    'in jeans and denim shirt',
    'in short white dress',
    'in long black dress',
    'in black dress and high heels',
    'topless',
    'brown coat',
    'black attire',
    'pajamas',
    'fishnet',
    'gerter belt'
    ]

# Envirompent details

enviropment = [
    'los angeles',
    'morming glow',
    'steam in air',
    'veinna',
    'sunny weather',
    'cloudly',
    'morning light',
    'evening light',
    'romantic light',
    'color light',
    ]

# Features

features = [
    'hot petite',
    'photo from behind',
    'dynamic pose',
    'backshot',
    'bottom view',
    'losely cropped',
    'view from back',
    'sideview',
    'looking away',
    'full body in view',
    'medium shot',
    'closeup shot',
    'panoramic',
    'front view',
    ]

# Photo style

photo_style = [
    'instagram',
    '2000s photo',
    '500px',
    'by Oliver Sin',
    ]

# Fix first params!


def construct_prompt():
    w1 = random.choice(prefix)
    w2 = random.choice(person)
    w3 = random.choice(hair)
    w4 = random.choice(body)
    w5 = random.choice(action)
    w6 = random.choice(location)
    w7 = random.choice(clothes)
    w8 = random.choice(enviropment)
    w9 = random.choice(features)
    w10 = random.choice(photo_style)
    
    prompt = f'{w1} {w2} with {w3} and {w4}, {w5} {w6}, {w7}, {w8}, {w9}, {w10}'
    
    return prompt

if __name__ == '__main__':
    print(construct_prompt())