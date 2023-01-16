import os
import json
import cv2
import numpy as np
import torch
from io import BytesIO
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import subprocess
from application import app
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from application.flask_dropzone import Dropzone
from transformers import pipeline
import ast
import moviepy.editor as mp
#from application.objective import ObjectiveTest
#from application.subjective import SubjectiveTest

basedir = os.path.abspath(os.path.dirname(__file__))

app.config.update(
    UPLOADED_PATH= os.path.join(basedir,'static/uploads'),
    DROPZONE_MAX_FILE_SIZE = 1024,
    DROPZONE_TIMEOUT = 5*60*1000)


dropzone = Dropzone(app)

def getFileName():
    uploaded = os.listdir('application/static/uploads')
    if len(uploaded) == 0:
        fileName = 'demo.mp4'
        path = 'application/static/demo.mp4'
        flag = 1
    else:
        fileName = os.listdir('application/static/uploads')[0]
        path = 'application/static/' + fileName
        flag = 0
    return fileName, path, flag

@app.route('/',methods=['POST','GET'])
def upload():
    print(len(request.values), request.values)
    if request.method == 'POST' and len(request.values) == 0 :
        files = os.listdir('application/static/uploads')
        for i in files:
            os.remove('application/static/uploads/'+i)
        f = request.files.get('file')
        f.save(os.path.join(app.config['UPLOADED_PATH'],f.filename))
    if request.method == 'POST' and request.form.get('action1') == 'SUBMIT':
        return searchPage(1)
    elif request.method == 'POST' and request.form.get('action1') == 'Try with demo video':
        return liveDetect()
    return render_template('index.html')

@app.route('/live',methods=['POST','GET'])
def liveDetect():
    fileName , path, flag = getFileName()
    if request.method == 'POST' and request.form.get('skipButton') == 'skip':
        return searchPage(1)
    return render_template('detection.html', demoOutput = fileName)

def detection(file, fileName, flag):
    if flag == 0:
        animals = [1,2,3,4]
        timeStamp = [5,6,7,8]
    else:
        animals = [1,2,3,4]
        timeStamp = [5,6,7,8]
        a,path,b = getFileName()
        #os.system("python application/yolov7-main/detect.py --weights application/models/detectionModel.pt --conf 0.5 --img-size 640 --source "+str(path)+" --no-trace" )
        with open('application/static/timestamps.txt')as f:
            lines = f.readlines()
        timeStamp = ast.literal_eval(lines[1])
        animals = ast.literal_eval(lines[0])
    return animals, timeStamp


@app.route('/search',methods=['POST','GET'])
def searchPage(count = 0):
    fileName , path, flag = getFileName()
    if count == 1:
        print("--------------------------------------------------------")
        animals, timeStamps = detection(fileName, fileName, flag)
    if flag == 0:
        fileName = "uploads/" + fileName

    if request.method == 'POST' and (request.form.get('action1') != 'SUBMIT' and request.form.get('action1') != 'Try with demo video' and request.form.get('skipButton') != 'skip'):
        targetAnimal = request.form.get('search')
        with open('application/static/timestamps.txt')as f:
            lines = f.readlines()
        timeStamps = ast.literal_eval(lines[1])
        animals = ast.literal_eval(lines[0])
        return playOriginal(timeStamps, targetAnimal, fileName)
    return render_template('search.html', animals=animals)

@app.route('/dashboard')
def playOriginal(timeStamps, targetAnimal, file):
    
    # transcript = voice_recognition("/application/static/marketplace_full.mp3")
    #summary = voice_recognition()
    
    n = 100
    summary = 'The white rhino weights over 2 tons and has a front horm of upto 5 feet. The horn can be used as a weapon. Made of keratine the same as claws and nails, rock solid and anchored deep inside its body. Its large neck hump is pure muscle for lifting its heavy head giving a powerful upward swing.'
    summary = summary.split()
    summary = [" ".join(summary[i:i+n]) for i in range(0, len(summary), n)]
    QA = 'What is the weight of a white rhino ?'
    QA = QA.split()
    QA = [" ".join(QA[i:i+n]) for i in range(0, len(QA), n)]

    '''
    transcript = audio_to_text("application/static/marketplace_full.mp3")
    obj, sub = test_generate(transcript)'''

    fileName, path, flag = getFileName()
    '''video = cv2.VideoCapture(path)
    currentFrame = 0

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    result = cv2.VideoWriter('application/static/trimmedVideo.mp4',-1,10, size)

    max = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(max)
    while(True):
        currentFrame += 1
        success, frame = video.read()
        if success == True and (currentFrame>=timeStamps[targetAnimal]["starting"] and currentFrame< timeStamps[targetAnimal]["ending"]): 
            result.write(frame)        
            
        elif currentFrame >= timeStamps[targetAnimal]["ending"]:
            break
    print(file)'''
    trimVid = 'trimmed.mp4'
    print(trimVid)

    return render_template('dashboardPage.html', originalVideo = fileName, trimmed = trimVid, QA = QA, summary = summary)

def audio_to_text(filename):
    FRAME_RATE = 16000
    CHANNELS=1
    model = Model("application/models/model")
    rec = KaldiRecognizer(model, FRAME_RATE)
    rec.SetWords(True)
    
    mp3 = AudioSegment.from_mp3(filename)
    mp3 = mp3.set_channels(CHANNELS)
    mp3 = mp3.set_frame_rate(FRAME_RATE)
    
    step = 45000
    transcript = ""
    for i in range(0, len(mp3), step):
        print(f"Progress: {i/len(mp3)}")
        segment = mp3[i:i+step]
        rec.AcceptWaveform(segment.raw_data)
        result = rec.Result()
        text = json.loads(result)["text"]
        transcript += text

    
    #cased = subprocess.check_output('python recasepunc/recasepunc.py predict recasepunc/checkpoint', shell=True, text=True, input=transcript)
    return transcript

def voice_recognition():
    n = 100
    fileName, path, flag = getFileName()

    my_clip = mp.VideoFileClip(path)
    my_clip.audio.write_audiofile('application/static/audio_result.mp3')


    transcript = audio_to_text("application/static/audio_result.mp3")

    split_tokens = transcript.split(" ")
    docs = []
    for i in range(0, len(split_tokens), 850):
        selection = " ".join(split_tokens[i:(i+850)])
        docs.append(selection)
    print(transcript)
    #summarizer = pipeline("summarization")
    summarizer = pipeline("summarization", model="t5-small")

    summaries = summarizer(docs)
    summary = "\n\n".join([d["summary_text"] for d in summaries])

    return [" ".join(summary[i:i+n]) for i in range(0, len(summary), n)]
    
@app.route('/test_generate', methods=["POST"])
def test_generate(inputText, noOfQues = 2):
	
    '''objective_generator = ObjectiveTest(inputText,noOfQues)
    question_list, answer_list = objective_generator.generate_test()
    objectivetestgenerate = zip(question_list, answer_list)


    subjective_generator = SubjectiveTest(inputText,noOfQues)
    question_list, answer_list = subjective_generator.generate_test()
    subjectivetestgenerate = zip(question_list, answer_list)


    return objectivetestgenerate, subjectivetestgenerate'''
    return ""


def question_generator():
     return ""