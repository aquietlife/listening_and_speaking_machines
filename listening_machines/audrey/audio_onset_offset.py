import pyaudio
import numpy as np
import soundfile as sf

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100#22050

p = pyaudio.PyAudio()

info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

onset_threshold = 0.025
offset_threshold = 0.009

RECORD = False

recorded_audio = []
print(" type ctrl+c to quit")

while True:
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.float32)

    rms = np.sqrt(audio_data ** 2).mean() # square each sample, take the mean, and then take the square root

    if rms > onset_threshold:
        print("onset!")
        RECORD = True
        recorded_audio.append(audio_data)

    elif rms < offset_threshold:
        RECORD = False
        if len(recorded_audio) > 0: # at least 1 second of audio
            print("offset!")
            print("recording")
            print(len(recorded_audio))
            print(recorded_audio)

            # concatenate all the recorded audio
            recorded_audio = np.concatenate(recorded_audio)

            # save the recorded audio to a file
            from datetime import datetime
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            sf.write(f'recorded_audio_{current_time}.wav', recorded_audio, RATE)
        recorded_audio = [] # reset recorded audio

stream.stop_stream()
stream.close()
p.terminate()