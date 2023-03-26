import erutils
import torch
from flask import Flask, send_file
from typing import Optional, Dict, List, Union, Tuple
from engine import config_model
from baseline import generate
import socket

COLOR: Optional[str] = erutils.Cp.RED
PORT: Optional[int] = 8080
MODEL_PATH: Optional[str] = r'E:\CGRModel-checkpoints'
SELECTED_DEVICE: Union[str, torch.device] = 'cpu'
IS_NSFW_ALLOWED: Optional[bool] = True
OUT_DIR: Optional[str] = 'out'
SIZE: Union[Tuple[int, int]] = (512, 512)
model = config_model(model_path=MODEL_PATH, device=SELECTED_DEVICE, nsfw_allowed=IS_NSFW_ALLOWED)
IP: Optional[str] = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)

erutils.fprint('MODEL LOADED *', color=COLOR)
erutils.fprint(f"START LISTENING ON : {IP} ", color=COLOR)
kwargs = dict(use_version=True, version='v4', use_realistic=False, size=SIZE, nsfw_allowed=IS_NSFW_ALLOWED,
              out_dir=OUT_DIR)


@app.route('/<prompt>')
def progress(prompt):
    erutils.fprint(f"|| GENERATING STARTED FOR PROMPT : [{prompt}] || ON DEVICE : {SELECTED_DEVICE}  ||")
    generate(prompt, model, **kwargs)
    return send_file(
        f'out/{prompt}.png',
        as_attachment=False,
        mimetype='image/png'
    )


if __name__ == "__main__":
    try:
        app.run(debug=True, host=IP, port=PORT)
    except ValueError as ve:
        erutils.fprint(ve)
