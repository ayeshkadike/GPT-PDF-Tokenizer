import tiktoken
from PyPDF2 import PdfReader, errors
import streamlit as st
from PIL import Image
from math import ceil
import tempfile
import os

st.title("GPT PDF Tokenizer")

#encoding
encoding = tiktoken.get_encoding("cl100k_base")

def decrypt_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)

    if pdf.is_encrypted:
        st.warning("This PDF is encrypted, please enter the password to unlock it.")
        password = st.text_input("Enter PDF Password:", type="password", key="pdf_password")

        if password:
            try:
                success = pdf.decrypt(password)
                if success == 0:
                    st.error("Incorrect password. Please try again.")
                    return None
                else:
                    st.success("PDF successfully decrypted.")
                    return pdf
            except errors.PdfReadError:
                st.error("Failed to decrypt the PDF. It may require a different password.")
                return None
        else:
            st.info("Waiting for password input...")
            return None
    else:
        return pdf

def extract_text(pdf):
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + '\n'
    return text

def extract_images(pdf):
    images = []
    count = 0

    for page in pdf.pages:
        for image_file_object in page.images:
            image_path = f"{count}_{image_file_object.name}"
            with open(image_path, "wb") as fp:
                fp.write(image_file_object.data)
                images.append(image_path)
                count += 1
    
    return images

def calculate_image_tokens(width: int, height: int):
    if width > 2048 or height > 2048:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width, height = 2048, int(2048 / aspect_ratio)
        else:
            width, height = int(2048 * aspect_ratio), 2048
            
    if width >= height and height > 768:
        width, height = int((768 / height) * width), 768
    elif height > width and width > 768:
        width, height = 768, int((768 / width) * height)

    tiles_width = ceil(width / 512)
    tiles_height = ceil(height / 512)
    total_tokens = 85 + 170 * (tiles_width * tiles_height)
    
    return total_tokens

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

# Process file
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Attempt to decrypt the PDF
    pdf = decrypt_pdf(tmp_file_path)

    # If decryption fails, stop execution
    if pdf is None:
        st.stop()

    # Extract text
    extracted_text = extract_text(pdf)
    
    # Display extracted text
    st.write("Extracted Text:")
    st.write(extracted_text)
    
    # Encode text
    text_tokens = encoding.encode(extracted_text)
    
    # Display text tokens
    st.write("Text Chunk Tokens:")
    st.write(text_tokens)
    
    # Extract and process images
    image_paths = extract_images(pdf)
    
    total_image_tokens = 0
    image_tokens = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
            tokens = calculate_image_tokens(width, height)
            image_tokens.append((image_path, tokens))
            total_image_tokens += tokens
            st.image(img, caption=f"{image_path} - Tokens: {tokens}")
    
    st.write("Image Tokens:")
    for image_path, tokens in image_tokens:
        st.write(tokens)

    # Calculate and display total tokens
    total_tokens = len(text_tokens) + total_image_tokens
    
    st.write("Total Tokens:")
    st.write(total_tokens)
    
    # Clean up temp files
    os.remove(tmp_file_path)
    for image_path in image_paths:
        os.remove(image_path)
