import base64
import tempfile

import numpy as np

from flask import current_app

from app.main import model, client
from app.main.util.inference import embed_utterance
from app.main.util.audio import preprocess_wav


def wav2temp(string):
    fp = tempfile.TemporaryFile()
    fp.write(string)
    fp.seek(0)

    return fp


def clone_voice(public_id, base64_file):
    decode_string = base64.b64decode(base64_file)

    fp = wav2temp(decode_string)
    out_wav = wav2temp(decode_string)

    encoder_wav = preprocess_wav(fp)
    embed = embed_utterance(model, encoder_wav)
    
    out_embed = tempfile.TemporaryFile()
    np.save(out_embed, embed, allow_pickle=True)

    out_embed.seek(0)

    client.upload_fileobj(out_embed, current_app.config['BUCKET_NAME'], f'{public_id}.npy')
    client.upload_fileobj(out_wav, current_app.config['BUCKET_NAME'], f'{public_id}.wav')

    return {
        'status': 'success',
        'message': 'Voice was cloned'
    }, 200
