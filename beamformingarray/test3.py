from Signal import Signal,Sine
from SignalGen import SignalGen
from AudioWriter import AudioWriter

sig=Sine(3000)
siggen=SignalGen(2,0.028)
signal=sig.generate_wave(8)
siggen.update_delays(45)
angled_signal=siggen.delay_and_gain(signal)
aw=AudioWriter()
aw.add_sample(angled_signal)
aw.write("./beamformingarray/Sine45deg28mm.wav",48000)