import numpy as np
from IOStream import IOStream
from beamformer import Beamformer
from AudioWriter import AudioWriter
from Preprocessor import Preprocessor
from time import time
from CrossCorrelator import CrossCorrelatior
from VAD import VAD
import soundfile as sf
from SignalGen import SignalGen
from Signal import Sine,Sawtooth,Chirp,Square
import Signal
import matplotlib.pyplot as plt

speech_database_path="C:\\Users\\arg\\Documents\\Datasets\\dev-clean.tar\\dev-clean\\LibriSpeech"
speech_database_subset="dev-clean"
speech_database_local=speech_database_path+"\\"+speech_database_subset
speakers=[]
chapters=[]
target_samplerate=48000
num_microphones=8
spacing=0.03
def open_chapter(chapter_num):
    print(chapter_num)
    chapter_string=speech_database_local+"\\"+chapters[chapter_num][1]+"\\"+chapters[chapter_num][0]+"\\"
    chapter_trans=open(chapter_string+chapters[chapter_num][1]+"-"+chapters[chapter_num][0]+".trans.txt") 
    chapter_splits=[]
    while(True):
        line=chapter_trans.readline()
        if(len(line)<=0):
            break
        arr=line.split(" ")
        chapter_splits.append(arr)
        chapter_splits[len(chapter_splits)-1][0]=chapter_string+arr[0]+".flac"
    return chapter_splits




def read_speakers():
    speakers_file=open(speech_database_path+"\\SPEAKERS.TXT")
    while(True):
        line =speakers_file.readline()
        if(len(line)<=0):
            break
        if( line[0]!=';'):
            arr=line.split('|')
            if(speech_database_subset in arr[2] ):
                for i in range(len(arr)):
                    arr[i]=arr[i].strip()
                speakers.append(arr)
            
def read_chapters():
    chapters_file=open(speech_database_path+"\\CHAPTERS.TXT")
    while(True):
        line =chapters_file.readline()
        if(len(line)<=0):
            break
        if( line[0]!=';'):
            arr=line.split('|')
            for i in range(len(arr)):
                    arr[i]=arr[i].strip()
            if(speech_database_subset in arr[3]):
                chapters.append(arr)
def generate_angles(num):
    angles=[]
    for i in range(num):
        angles.append(int(np.random.rand()*180)) # IMPORTANT!!!!: THIS IS JUST FOR SINGLE LINEAR ARRAY NEED THE TRANFORMATION FOR 2 ARRAY SYSTEM

    return angles

##########################################################################################

# Synthetic Signal Tests
# num_single_source_synthetic_tests=0
# synthetic_tests_length=5
# min_frequency=100
# max_frequency=1000
# synthetic_single_source_rms=[]
# synthetic_single_source_signed_mean=[]
# synthetic_single_source_unsigned_mean=[]
# synthetic_single_source_angles=generate_angles(num_single_source_synthetic_tests)
# sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
# synthetic_single_source_freq=[]
# for i in range(num_single_source_synthetic_tests):
#     print(i)
#     doa_diff=[]
#     beamformer=Beamformer(num_microphones,spacing,target_samplerate)
#     freq=np.random.randint(min_frequency,max_frequency)
#     synthetic_single_source_freq.append(freq)
#     if(i+1==num_single_source_synthetic_tests):
#         signal=Chirp(start_freq=min_frequency,end_freq=max_frequency,sample_rate=target_samplerate)
#     elif(i%3==0):
#         signal=Sine(frequency=freq,sample_rate=target_samplerate)
#     elif(i%3==1):
#         signal=Sawtooth(frequency=freq,sample_rate=target_samplerate)
#     elif(i%3==2):
#         signal=Square(frequency=freq,sample_rate=target_samplerate)
    
#     sig_data=signal.generate_wave(synthetic_tests_length)
#     sig_gen.update_delays(synthetic_single_source_angles[i])
#     angled_sig_data=sig_gen.delay_and_gain(sig_data)
#     io=IOStream()
#     io.arrToStream(angled_sig_data,target_samplerate)
#     aw=AudioWriter()
#     while(not io.complete()):
#         frame=io.getNextSample()
#         aw.add_sample(beamformer.beamform(frame))
#         doa_diff.append(synthetic_single_source_angles[i]-beamformer.doa)
#         # print(doa_diff)
    
#     doa_diff_arr=np.array(doa_diff)
#     synthetic_single_source_rms.append(np.sqrt(np.mean(doa_diff_arr**2)))
#     synthetic_single_source_signed_mean.append(np.mean(doa_diff_arr))
#     synthetic_single_source_unsigned_mean.append(np.mean(np.abs(doa_diff_arr)))
# print(np.mean(synthetic_single_source_unsigned_mean))
# print(np.mean(synthetic_single_source_signed_mean))
# print(np.mean(synthetic_single_source_rms))

#Single Source Attenuation Test
# read_speakers()
# read_chapters()
# attenuation_test_length=0.5
# min_angle=0
# max_angle=180
# source_angle=0
# min_frequency=300
# max_frequency=3000
# sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
# chapter_splits=open_chapter(0)
# speech,samplerate=sf.read(chapter_splits[0][0])
# print(chapter_splits[0][0])
# speech=np.reshape(speech,(-1,1))
# interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))

# speech=interpolator.process(speech)
# sig=Chirp(min_frequency,max_frequency)
# sig_data=speech
# # sig_data=sig.generate_wave(attenuation_test_length)
# sig_gen.update_delays(source_angle)
# angled_sig_data=sig_gen.delay_and_gain(sig_data)
# angles=[]
# rms_data=[]
# for i in range(min_angle,max_angle):
#     print(i)
#     angles.append(i)
#     beamformer=Beamformer(num_microphones,spacing,target_samplerate)
#     beamformer.toggle_doa_lock()
#     beamformer.update_delays(i)
#     attenuated_sig_data=beamformer.beamform(angled_sig_data)
#     rms=np.sqrt(np.mean(attenuated_sig_data**2))
#     rms_data.append(rms)    
# angles_arr=np.array(angles)
# rms_data_arr=(np.array(rms_data))
# rms_data_arr=rms_data_arr/max(rms_data_arr)
# plt.figure(figsize=(10,4),dpi=100)
# ax = plt.subplot(131)
# ax.plot(angles_arr,rms_data_arr)
# ax.grid(True)
# ax.set_xlim([0, 180])
# ax.set_xlabel("Theta (Degrees)")
# ax.set_ylabel("Amplitude (Decibels)")
# plt.axvline(x = source_angle, color = 'b', label = 'axvline - full height')
# plt.show()

# Multiple Source Attenuation Test
# attenuation_test_length=0.5
# min_angle=0
# max_angle=180
# source1_angle=70
# distance1=1
# source2_angle=110
# distance2=1
# min_frequency=100
# max_frequency=3000
# sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
# # sig1=Chirp(min_frequency,max_frequency)
# sig1=Chirp(start_freq=min_frequency,end_freq=max_frequency,sample_rate=target_samplerate)
# sig1_data=sig1.generate_wave(attenuation_test_length)
# sig_gen.update_delays(source1_angle)
# sig_gen.update_gains(distance1)
# angled_sig1_data=sig_gen.delay_and_gain(sig1_data)
# sig2=Sine(frequency=max_frequency/2+min_frequency/2,sample_rate=target_samplerate)
# sig2_data=sig2.generate_wave(attenuation_test_length)
# sig_gen.update_delays(source2_angle)
# sig_gen.update_gains(distance2)
# angled_sig2_data=sig_gen.delay_and_gain(sig2_data)
# summed_data=Signal.sum_signals(angled_sig1_data,angled_sig2_data)

# angles=[]
# rms_data=[]
# for i in range(min_angle,max_angle):
#     print(i)
#     angles.append(i)
#     beamformer=Beamformer(num_microphones,spacing,target_samplerate)
#     beamformer.toggle_doa_lock()
#     beamformer.update_delays(i)
#     attenuated_sig_data=beamformer.beamform(summed_data)
#     rms=np.sqrt(np.mean(attenuated_sig_data**2))
#     rms_data.append(rms)    
# angles_arr=np.array(angles)
# rms_data_arr=(np.array(rms_data))
# rms_data_arr=rms_data_arr/max(rms_data_arr)
# plt.figure(figsize=(10,4),dpi=100)
# ax = plt.subplot(131)
# ax.plot(angles_arr,rms_data_arr)
# ax.grid(True)
# ax.set_xlim([0, 180])
# ax.set_xlabel("Theta (Degrees)")
# ax.set_ylabel("Amplitude (Decibels)")
# plt.axvline(x = source1_angle, color = 'b', label = 'axvline - full height')
# plt.axvline(x = source2_angle, color = 'r', label = 'axvline - full height')
# plt.show()

#Single Source Tests
read_speakers()
read_chapters()
num_microphones=8
spacing=0.03
num_single_source_tests=10
ang_iter=0
chapter_ind=0
chapter_splits=open_chapter(chapter_ind)
split_iterator=0
interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
single_source_rms=[]
single_source_signed_mean=[]
single_source_unsigned_mean=[]
single_source_angles=generate_angles(num_single_source_tests) # IMPORTANT!!!!: THIS IS JUST FOR SINGLE LINEAR ARRAY NEED THE TRANFORMATION FOR 2 ARRAY SYSTEM
print(single_source_angles)
for i in range(num_single_source_tests):
    print(i)
    doa_diff=[]
    beamformer=Beamformer(num_microphones,spacing,target_samplerate)
    if chapter_ind>=len(chapters):
        chapter_ind=0
    if split_iterator>=len(chapter_splits):
        split_iterator=0
        chapter_ind+=1
        chapter_splits=open_chapter(chapter_ind)
    speech,samplerate=sf.read(chapter_splits[split_iterator][0])
    speech=np.reshape(speech,(-1,1))
    speech=interpolator.process(speech)
    sig_gen.update_delays(single_source_angles[i])
    angled_speech=sig_gen.delay_and_gain(speech)
    io=IOStream()
    io.arrToStream(angled_speech,target_samplerate)
    aw=AudioWriter()
    while(not io.complete()):
        frame=io.getNextSample()
        aw.add_sample(beamformer.beamform(frame))
        doa_diff.append(single_source_angles[i]-beamformer.doa)
        # print(doa_diff)
    split_iterator+=1
    doa_diff_arr=np.array(doa_diff)
    single_source_rms.append(np.sqrt(np.mean(doa_diff_arr**2)))
    single_source_signed_mean.append(np.mean(doa_diff_arr))
    single_source_unsigned_mean.append(np.mean(np.abs(doa_diff_arr)))
print(np.mean(single_source_unsigned_mean))
print(np.mean(single_source_signed_mean))
print(np.mean(single_source_rms))
    
    



def add_noise(speech,noise):
    return speech


