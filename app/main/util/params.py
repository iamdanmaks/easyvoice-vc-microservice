
mel_window_length = 25
mel_window_step = 10
mel_n_channels = 40


sampling_rate = 16000
partials_n_frames = 160
inference_n_frames = 80


vad_window_length = 30
vad_moving_average_width = 8
vad_max_silence_length = 6


audio_norm_target_dBFS = -30

# Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3

learning_rate_init = 1e-4
speakers_per_batch = 64
utterances_per_speaker = 10
