import flask
from flask import Flask, request
import torch
import io

from model.sdf_net import SDFNet
from model import LATENT_CODE_SIZE

sdf_net = SDFNet()
current_model = None

app = Flask(__name__)

@app.route('/voxels')
def get_voxels():
    global current_model
    resolution = request.args.get("resolution", 32, type=int)
    model_filename = request.args.get("model_filename", None)

    if current_model is None or current_model != model_filename:
        model = torch.load(model_filename)
        current_model = model_filename
        sdf_net.load_state_dict(model)

    latent_code = torch.zeros(LATENT_CODE_SIZE, device=sdf_net.device)
    voxels = sdf_net.get_voxels(latent_code, voxel_resolution=resolution, sphere_only=False)
    sdf_bytes = voxels.flatten().tobytes()

    return flask.Response(io.BytesIO(sdf_bytes), mimetype='application/octet-stream')

app.run()
