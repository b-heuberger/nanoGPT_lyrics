#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 00:24:27 2023

@author: benjaminheuberger
"""

import os
import pickle
import requests
import numpy as np
import yaml 
import argparse

artist = 'Bob Dylan'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process configuration')
    parser.add_argument('--artist', type=str, help='Name of the artist for data preparation')
    args = parser.parse_args()

    if args.artist:
        # Replace the 'out_dir' variable with the artist's name
        out_dir = args.artist.replace(' ', '_')
        print(f"Using artist: {args.artist}")
        artist = args.artist
    else:
        print("No artist provided. Using default configuration: Bob Dylan.")

artist_with_underscore = artist.replace(' ', '_')
yaml_file = artist_with_underscore + '_Lyrics.yaml'
input_file_path = os.path.join(os.path.dirname(__file__), yaml_file)

# Load the YAML file
with open(input_file_path, "r") as yaml_file:
    lyrics = yaml.safe_load(yaml_file)  
    
lyrics = lyrics[artist]

data = ""

for song_lyrics in lyrics.values():
    data += song_lyrics + "\n"  # Add each song's lyrics, consider adding a newline for separation

# If you want to remove the last newline character
data = data.rstrip("\n")
n = len(data)

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)