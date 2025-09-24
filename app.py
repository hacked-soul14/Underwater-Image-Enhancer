import os
import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_file
from torchvision import transforms
from hybrid import HybridEnhancer, post_process_image

app = Flask(__name__)

# Load the model once at startup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HybridEnhancer().to(device)
model.load_state_dict(torch.load('hybrid_enhancer1.pth', map_location=device))
model.eval()

# Transform for input image
input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def enhance_image(input_img):
    img = input_transform(input_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
    output = output.squeeze(0).cpu()
    # Unnormalize and post-process
    output = output * 0.5 + 0.5
    output_np = output.clamp(0, 1).numpy().transpose(1, 2, 0)
    output_np = post_process_image(output_np)
    output_img = Image.fromarray((output_np * 255).astype(np.uint8))
    return output_img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        input_img = Image.open(file.stream).convert('RGB')
        enhanced_img = enhance_image(input_img)
        # Save to buffer
        buf = io.BytesIO()
        enhanced_img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
