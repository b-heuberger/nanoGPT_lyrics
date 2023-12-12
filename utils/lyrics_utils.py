#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:26:27 2023

@author: benjaminheuberger
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(out_dir=str):
    """
    Plots training and validation loss across iterations 

    Args:
    out-dir (str): The out directory that stores the model checkpoint file

    Returns:
    None, generates plot 
        
    """

    # Define the full path for the CSV file
    log_path = os.path.join(out_dir, 'metrics_log.csv')

    # Load the CSV file into a DataFrame from the updated path
    df = pd.read_csv(log_path)

    # Clean up the 'tensor' and parentheses from the columns containing such values
    def clean_tensor(text):
        return float(text.split('(')[1].split(')')[0])

    # Apply the clean-up function to the columns with 'tensor' values
    df['train/loss'] = df['train/loss'].apply(clean_tensor)
    df['val/loss'] = df['val/loss'].apply(clean_tensor)

    # Plotting the metrics over iterations
    plt.figure(figsize=(10, 6))

    plt.plot(df['iter'], df['train/loss'], label='Train Loss')
    plt.plot(df['iter'], df['val/loss'], label='Validation Loss')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.show()
    

def postprocessing_lyrics(lyrics):
    """
    Improves the formatting and structure of lyric output to better reflect an actual song.

    Args:
    lyrics (str): A string of generated lyrics 

    Returns:
    str: Cleaned lyrics 
    """

    # Split the lyrics by new lines
    lines = lyrics.split('\n')
    verse_count = 0
    
    # Criteria 1: Insert blank lines where missing between a verse header and prior paragraph
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("[Verse"):
            verse_count += 1
            if i > 0 and lines[i - 1] != '' and not lines[i - 1].startswith("[Verse"):
                lines.insert(i, '')

    # Criteria 2: Ensure verse numbers count up and never down          
    expected_verse = 1
    for idx, line in enumerate(lines):
        if line.startswith("[Verse") and ']' in line:
            try: 
              verse_num = int(line.split(']')[0][7:])
              if verse_num < expected_verse:
                  # Truncate at the last complete verse before the violation
                  lines = lines[:idx]
                  break
              expected_verse += 1
            except ValueError:
              continue 

    # Criteria 3: Truncate at the last complete verse
    for i in range(len(lines) - 1, -1, -1):
        if lines[i] == '':
            lines = lines[:i]
            break

    # Reconstruct cleaned lyrics
    cleaned_lyrics = '\n'.join(lines)
    return cleaned_lyrics

