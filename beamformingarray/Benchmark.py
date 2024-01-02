import numpy as np
from IOStream import IOStream
from beamformer import Beamformer
from AudioWriter import AudioWriter
from Preprocessor import Preprocessor
from time import time
from CrossCorrelator import CrossCorrelatior
from VAD import VAD
import soundfile as sf
speech_database_path="C:\\Users\\arg\\Documents\\Datasets\\dev-clean.tar\\dev-clean\\LibriSpeech"
speech_database_subset="dev-clean"
speech_database_local=speech_database_path+"\\"+speech_database_subset
speakers=[]

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
            
# print(speakers)
chapters_file=open(speech_database_path+"\\CHAPTERS.TXT")
chapters=[]
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
         
# print(chapters)
# FOR EACH CHAPTER
chapter_string=speech_database_local+"\\"+chapters[0][1]+"\\"+chapters[0][0]+"\\"
chapter_trans=open(chapter_string+chapters[0][1]+"-"+chapters[0][0]+".trans.txt") 
chapter_splits=[]
while(True):
    line=chapter_trans.readline()
    if(len(line)<=0):
        break
    arr=line.split(" ")
    chapter_splits.append(arr)

print(chapter_trans.readline())                   
speech,samplerate=sf.read(chapter_string+chapter_splits[0][0]+".flac")
speech=np.reshape(speech,(-1,1))

# print(samplerate)
print(speech.shape)
target_samplerate=48000
interpolator=Preprocessor(mirrored=False,interpolate=int(1+np.ceil(target_samplerate/samplerate)))
speech=interpolator.process(speech)
io=IOStream()
io.arrToStream(speech,target_samplerate)

print(speech.shape[0])