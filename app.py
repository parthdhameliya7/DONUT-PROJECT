import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel, DonutProcessor, VisionEncoderDecoderConfig
import numpy as np
import pandas as pd
import re
import json
import cv2
from PIL import Image
import time
from src import params
from src import  DonutModel, Transforms
import plotly.graph_objects as go

# Set page config for wide mode and add a title
st.set_page_config(page_title="DONUT: OCR-Free Document Understanding 🤖", layout="wide")

# Apply custom CSS for dark mode
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
    }
    .stFileUploader>label {
        color: #4CAF50;
    }
    .stImage {
        border: 2px solid #4CAF50;
    }
    .stTextInput>div>input {
        color: white;
        background-color: #333333;
        border-color: #333333;
    }
    .stTextInput>div>input:focus {
        border-color: #4CAF50;
    }
    .stMarkdown p {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model():
    processor = DonutProcessor.from_pretrained('naver-clova-ix/donut-base')
    processor.tokenizer.eos_token = params['end_token']
    processor.tokenizer.add_tokens(params['special_tokens'] + [params['start_token']] + [params['end_token']])
    params['vocab_size'] = len(processor.tokenizer)

    config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
    config.encoder.image_size = list(params['image_size'])
    config.decoder.max_length = params['max_length']

    donut = VisionEncoderDecoderModel(config=config)
    donut.decoder.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=16)
    donut.config.pad_token_id = processor.tokenizer.pad_token_id
    donut.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([params['start_token']])[0]

    model = DonutModel(donut, processor)
    model.load_state_dict(torch.load('/Users/parth/Desktop/DONUT-PROJECT/models/donut_model_0_0_1.pt', map_location='cpu')['state_dict'])
    model.to('cpu')
    return model, processor

model, processor = load_model()

@torch.no_grad()
def get_predictions(image):
    """
    Generates a prediction from a given image using the model.

    Args:
    - image (Tensor): The input image tensor.

    Returns:
    - str: The decoded sequence prediction.
    """
    # Step 1: Initialize decoder input with the start token
    decoder_input_ids = torch.full(
        (1, 1), 
        model.donut.config.decoder_start_token_id,
        device = 'cpu'
    )

    # Step 2: Generate output using the model
    output = model.donut.generate(
        image,
        decoder_input_ids=decoder_input_ids,
        max_length=params['max_length'],
        early_stopping=True, 
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True, 
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True
    )

    seq = processor.tokenizer.decode(output.sequences[0])
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  
    seq = processor.token2json(seq)

    return seq

# Streamlit app
st.title("DONUT : OCR-Free Document Understanding Model 🤖")
st.markdown("### Upload an image and watch the magic happen!")

# File uploader with a different style
uploaded_file = st.file_uploader("Choose an image...", type="png", help="Upload a PNG image for analysis.")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    train_augs, valid_augs = Transforms(image_size=params['image_size']).get_transforms()
    image_np = valid_augs(image=image_np)['image'] / 255.0
    image_np = image_np.unsqueeze(0)

    seq = get_predictions(image_np.to('cpu'))

    seq_json = json.dumps(
        seq,
        sort_keys=True,
        indent=4,
        separators=(',', ': ')
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        placeholder = st.empty()
        for I in range(1, len(seq_json) + 1):
            if I == len(seq_json):
                text_with_cursor = seq_json
            else:
                text_with_cursor = seq_json[:I] + "●"
            placeholder.code(text_with_cursor, language="json")
            time.sleep(0.01)  # Adjust the speed of the streaming effect

# Add some footer information
st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <p>Developed by Parth Dhameliya & Fenil Savani.</p>
        <p><a href='https://github.com/parthdhameliya7/DONUT-PROJECT' target='_blank'>GitHub Repository</a></p>
    </div>
    """, unsafe_allow_html=True)