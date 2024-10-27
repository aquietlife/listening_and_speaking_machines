import pyaudio
import numpy as np
import soundfile as sf
import torch.nn as nn

import torchaudio
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import librosa
import torch as t

mel_freq_bins = 128
time_steps = 89
longest_audio_file_length = 17647

# model architecture
class ConvModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, mel_freq_bins, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of the flattened features
        self.flat_features = mel_freq_bins * (mel_freq_bins // 8) * (time_steps // 8)
        
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, 1, 128, 366)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.flat_features)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# get our device
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load the model
conv_model_loaded = ConvModel()
conv_model_loaded.load_state_dict(t.load(f'model_weights/audrey_model_weights_2024-10-26.pth', map_location=device))


CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050

p = pyaudio.PyAudio()

info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

#for i in range(0, numdevices):
#    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

onset_threshold = 0.025
offset_threshold = 0.005

RECORD = False

recorded_audio = []
print(" type ctrl+c to quit")

phone_number = []

while True:
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.float32)

    rms = np.sqrt(audio_data ** 2).mean() # square each sample, take the mean, and then take the square root

    if rms > onset_threshold:
        print("onset!")
        #print("recording")
        RECORD = True
        recorded_audio.append(audio_data)

    elif rms < offset_threshold:
        RECORD = False
        if len(recorded_audio) > 0: # at least 1 second of audio
            print("processing audio")
            #print("offset!")
            #print("listening")
            #print(len(recorded_audio))
            #print(recorded_audio)

            # concatenate all the recorded audio
            recorded_audio = np.concatenate(recorded_audio)

            # save the recorded audio to a file
            from datetime import datetime
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            #sf.write(f'recorded_audio_{current_time}.wav', recorded_audio, RATE)
            #sf.write(f'test.wav', recorded_audio, RATE)

            #audio, sample_rate = librosa.load('test.wav', sr=22050)

            current_size = len(recorded_audio)
            pad_size = longest_audio_file_length - current_size
            left_pad = pad_size // 2
            right_pad = pad_size - left_pad
            padded_audio = np.pad(recorded_audio, (left_pad, right_pad), mode='constant')
            #sf.write('test_padded.wav', padded_audio, RATE)

            #audio, sample_rate = librosa.load('test_padded.wav', sr=RATE)

            #plt.figure(figsize=(10, 4))
            #plt.plot(audio)
            #plt.title("Recorded Audio Waveform")
            #plt.show()

            #display(Audio(audio, rate=RATE))
            audio = t.tensor([np.array(padded_audio)])

            # display spectrogram
            spec = torchaudio.transforms.MelSpectrogram()(audio)

            # Pass spec into model
            outputs = conv_model_loaded(spec)
            prediction = outputs.argmax().item()
            #print(outputs)
            # Print the prediction
            print("Model prediction:", prediction)
            phone_number.append(prediction)

            if len(phone_number) == 10:
                print("phone number:", phone_number)
                phone_number = []
        
        recorded_audio = [] # reset recorded audio

stream.stop_stream()
stream.close()
p.terminate()



