import librosa
import matplotlib.pyplot as plt
import numpy as np

#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


# def int_or_str(text):
#     """Helper function for argument parsing."""
#     try:
#         return int(text)
#     except ValueError:
#         return text


# parser = argparse.ArgumentParser(add_help=False)
# parser.add_argument(
#     '-l', '--list-devices', action='store_true',
#     help='show list of audio devices and exit')
# args, remaining = parser.parse_known_args()
# if args.list_devices:
#     print(sd.query_devices())
#     parser.exit(0)
# parser = argparse.ArgumentParser(
#     description=__doc__,
#     formatter_class=argparse.RawDescriptionHelpFormatter,
#     parents=[parser])
# parser.add_argument(
#     'channels', type=int, default=[1,2], nargs='*', metavar='CHANNEL',
#     help='input channels to plot (default: the first)')
# parser.add_argument(
#     '-d', '--device', type=int_or_str,
#     help='input device (numeric ID or substring)')
# parser.add_argument(
#     '-w', '--window', type=float, default=400, metavar='DURATION',
#     help='visible time slot (default: %(default)s ms)')
# parser.add_argument(
#     '-i', '--interval', type=float, default=30,
#     help='minimum time between plot updates (default: %(default)s ms)')
# parser.add_argument(
#     '-b', '--blocksize', type=int, help='block size (in samples)')
# parser.add_argument(
#     '-r', '--samplerate', type=float, help='sampling rate of audio device')
# parser.add_argument(
#     '-n', '--downsample', type=int, default=1, metavar='N',
#     help='display every Nth sample (default: %(default)s)')
# args = parser.parse_args(remaining)
# if any(c < 1 for c in args.channels):
#     parser.error('argument CHANNEL: must be >= 1')
# mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
# q = queue.Queue()


# def audio_callback(indata, frames, time, status):
#     """This is called (from a separate thread) for each audio block."""
#     if status:
#         print(status, file=sys.stderr)
#     # Fancy indexing with mapping creates a (necessary!) copy:
#     q.put(indata[::args.downsample, mapping])


# def update_plot(frame):
#     """This is called by matplotlib for each plot update.

#     Typically, audio callbacks happen more frequently than plot updates,
#     therefore the queue tends to contain multiple blocks of audio data.

#     """
#     global plotdata
#     while True:
#         try:
#             data = q.get_nowait()
#         except queue.Empty:
#             break
#         shift = len(data)
#         plotdata = np.roll(plotdata, -shift, axis=0)
#         plotdata[-shift:, :] = data

  
#     for column, line in enumerate(lines):
#         line.set_ydata(plotdata[:, column])
#     return lines


# try:
#     if args.samplerate is None:
        
#         device_info = sd.query_devices(args.device, 'input')
#         print(device_info)
#         args.samplerate = device_info['default_samplerate']

#     length = int(args.window * args.samplerate / (1000 * args.downsample))
#     plotdata = np.zeros((length, len(args.channels)))

#     fig, ax = plt.subplots()
#     lines = ax.plot(plotdata)
#     if len(args.channels) > 1:
#         ax.legend([f'channel {c}' for c in args.channels],
#                   loc='lower left', ncol=len(args.channels))
#     ax.axis((0, len(plotdata), -1, 1))
#     ax.set_yticks([0])
#     ax.yaxis.grid(True)
#     ax.tick_params(bottom=False, top=False, labelbottom=False,
#                    right=False, left=False, labelleft=False)
#     fig.tight_layout(pad=0)

#     stream = sd.InputStream(
#         device=args.device, channels=max(args.channels),
#         samplerate=args.samplerate, callback=audio_callback)
#     ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
#     with stream:
#         plt.show()
# except Exception as e:
#     parser.exit(type(e).__name__ + ': ' + str(e))
#y, sr = librosa.load("babble_10.wav", duration=10)
y, sr = librosa.load(librosa.ex('libri3'), duration=3)
D = np.abs(librosa.stft(y))
times = librosa.times_like(D)
fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[0])
ax[0].set(title='Power spectrogram')
ax[0].label_outer()
onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         aggregate=np.mean,
                                         fmax=8000, n_mels=256)
print((np.mean(onset_env)))
ax[1].plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,
           label='Median (custom mel)')
C = np.abs(librosa.cqt(y=y, sr=sr))
onset_env = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))
ax[1].plot(times, onset_env / onset_env.max(), alpha=0.8,
         label='Mean (CQT)')
ax[1].legend()
ax[1].set(ylabel='Normalized strength', yticks=[])
plt.show()