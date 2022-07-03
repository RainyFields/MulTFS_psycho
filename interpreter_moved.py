#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.1.2),
    on May 05, 2022, at 12:16
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019)
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195.
        https://doi.org/10.3758/s13428-018-01193-y

"""

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
import csv
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions

from psychopy.hardware import keyboard
import pandas as pd
import random
from PIL import Image
import json

import argparse

root = os.getcwd()

curr_sys = os.name

is_posix = False
if curr_sys == "posix":
    is_posix = True

def LoadFromFile(namespace,):
    myfile = "config.json"
    file = open(os.path.join(root, myfile))
    data = json.load(file)
    return data[namespace]

parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str, required = False, default = LoadFromFile("task"))
parser.add_argument('--img_path', type = str, required = False, default = LoadFromFile("img_path"))
parser.add_argument('--n_trials', type = int, required = False, default = LoadFromFile("n_trials"))
parser.add_argument('--instruction_duration', type = float, required = False, default = LoadFromFile("instruction_duration"))
parser.add_argument('--frame_duration', type = float, required = False, default = LoadFromFile("frame_duration"))
parser.add_argument('--ISI', type = float, required = False, default = LoadFromFile("ISI"))
args = parser.parse_args()


if args.task == "1-back_identity":
    if is_posix == True:
        task_dir = os.path.join(root, "1back_identity")
    else:
        task_dir = os.path.join(root, "1back_identity", "1back_identity")

ins_duration = args.instruction_duration
frame_duration = args.frame_duration
ISI = args.ISI

psychopyVersion = '2022.1.2'
expName = args.task 
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit() 
expInfo['date'] = data.getDateStr()  
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

if is_posix == True:
    filename = './' + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    shortened_filename = './' + os.sep + "key_presses_" + expInfo['participant']
else:
    filename = os.path.dirname(os.path.abspath(__file__)) + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    shortened_filename = os.path.dirname(os.path.abspath(__file__)) + os.sep + "key_presses_" + expInfo['participant']


if is_posix == True:
    thisExp = data.ExperimentHandler(name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='./interpreter_moved.py',
        savePickle=True, saveWideText=True,
        dataFileName=filename)
else:
    thisExp = data.ExperimentHandler(name=expName, version='',
                                     extraInfo=expInfo, runtimeInfo=None,
                                     originPath=os.path.abspath(__file__),
                                     savePickle=True, saveWideText=True,
                                     dataFileName=filename)

logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING) 

endExpNow = False  
frameTolerance = 0.001  

win = visual.Window(
    size=(500, 500), fullscr=False, screen=0,
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
    blendMode='avg', useFBO=True,
    units='height')

filelist = []
rootdir = task_dir
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if(os.path.isdir(d)):
        filelist.append(d)
filelist.pop()
filelist.remove(filelist[0])

random_index_list = []
for i in range(int(args.n_trials)):
    rand_gen = random.randint(0, len(filelist)-1)
    while (rand_gen in random_index_list):
        rand_gen = random.randint(0, len(filelist)-1)
    random_index_list.append(rand_gen)
random_trials = []
for j in range(len(random_index_list)):
     random_trials.append(filelist[random_index_list[j]])
frame_log = {}
num_total_epochs = 0
current_frame_num = 0
for a in range(int(args.n_trials)):
    task_number = str(random_trials[a])
    folder_number = task_number
    task_path = os.path.join(f'{task_number}', 'compo_task_example')

    fobj = open(task_path, "r")
    content = json.load(fobj)

    trial_total_epochs = content['epochs']
    objects = content['objects']
    df = pd.read_pickle(os.path.join(root, "MULTIF_5_stim","MULTFS_5_stim.pkl", ))
    instr = content['instruction']
    answers = content['answers']
    img_log = {}
    frame_log[current_frame_num] = [True, None, None, None, None, None, None, None, a, instr]
    current_frame_num += 1
    for i in range(len(objects)):
        obj = objects[i]
        location = (obj['location'][1] - 0.5 , -(obj['location'][0] - 0.5))
        ctg_mod = int(obj['category'])
        obj_mod = int(obj['object'])
        ang_mod = int(obj['view angle'])
        img_epoch = int(obj['epochs'])
        is_distractor = obj['is_distractor']
        curr_answer = answers[i]
        is_instr = False

        ref = int(df.loc[(df['category'] == ctg_mod) & (df['obj_mod'] == obj_mod) & (df['ang_mod'] == ang_mod)].sample()['ref'])

        img_path = os.path.join(root + args.img_path, f'{ref}','image.png')
        im = Image.open(img_path, 'r')
        img_log[img_epoch] = [is_instr, im, location, ctg_mod, obj_mod, ang_mod, is_distractor, curr_answer, a, None]
    for j in range(trial_total_epochs):
        if img_log.get(j) == None:
            img_log[j] = [False, None, None, None, None, None, None, None, a, None]
    for k in range(len(img_log)):
        frame_log[current_frame_num] = img_log.get(k)
        current_frame_num += 1
    num_total_epochs = num_total_epochs + trial_total_epochs + 1

defaultKeyboard = keyboard.Keyboard(backend='iohub')

trialClock = core.Clock()
stimulus = visual.ImageStim(
    win=win,
    name='stimulus',
    image=None, mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=(0.2, 0.2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
instructions = visual.TextStim(win=win, name='instructions',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.02, wrapWidth=0.4, ori=0.0,
    color='white', colorSpace='rgb', opacity=None,
    languageStyle='LTR',
    depth=-1.0);
fixation_point = visual.ShapeStim(
    win=win, name='fixation_point', vertices='cross',
    size=[0.05, 0.05],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-2.0, interpolate=True)
key_resp = keyboard.Keyboard()

globalClock = core.Clock() 
routineTimer = core.CountdownTimer()  

epochs = data.TrialHandler(nReps=num_total_epochs + 1, method='sequential',
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='epochs')
thisExp.addLoop(epochs) 
thisEpoch = epochs.trialList[0]  

if thisEpoch != None:
    for paramName in thisEpoch:
        exec('{} = thisEpoch[paramName]'.format(paramName))
frame_num = 0
key_responses_list = [['Frame Num', 'Key', 'Time']]
key_logger = csv.writer(open(shortened_filename + '.csv', 'w'), lineterminator = '\n')
key_logger.writerow(["Frame Num", "Key", "Response Time", "Global Time", "Correct Answer?"])
for thisEpoch in epochs:
    currentLoop = epochs

    if thisEpoch != None:
        for paramName in thisEpoch:
            exec('{} = thisEpoch[paramName]'.format(paramName))

    continueRoutine = True
    routineTimer.add(100000.000000)

    current_frame = frame_log.get(frame_num)
    if(current_frame[0] == None or current_frame[0] == False ):
        instructions.setText("")
        fixation_point.setSize((0.05, 0.05))
        if(current_frame[1] == None):
            stimulus.setImage(None)
        else:
            stimulus.setPos(current_frame[2])
            stimulus.setImage(current_frame[1])
        if(current_frame[7] != "null"):
            fixation_point.setSize((0, 0))
        else:
            fixation_point.setSize((0.05, 0.05))
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
    else:
        stimulus.setImage(None)
        instructions.setText(current_frame[-1] + "\n Press 'Space' to continue." + "\n Trial number " + str(current_frame[8] + 1) )
        fixation_point.setSize((0, 0))
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []

    frame_num += 1
    trialComponents = [stimulus, instructions, fixation_point, key_resp]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trialClock.reset(-_timeToFirstFrame) 
    frameN = -1

    while continueRoutine and routineTimer.getTime() > 0:

        t = trialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  

        if stimulus.status == NOT_STARTED and tThisFlip >= ISI-frameTolerance:

            stimulus.frameNStart = frameN  
            stimulus.tStart = t  
            stimulus.tStartRefresh = tThisFlipGlobal 
            win.timeOnFlip(stimulus, 'tStartRefresh') 
            stimulus.setAutoDraw(True)
        if stimulus.status == STARTED:

            if tThisFlipGlobal > stimulus.tStartRefresh + frame_duration-frameTolerance:

                stimulus.tStop = t 
                stimulus.frameNStop = frameN 
                win.timeOnFlip(stimulus, 'tStopRefresh') 
                stimulus.setAutoDraw(False)

        if instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            instructions.frameNStart = frameN  
            instructions.tStart = t  
            instructions.tStartRefresh = tThisFlipGlobal  
            win.timeOnFlip(instructions, 'tStartRefresh')
            instructions.setAutoDraw(True)
        if instructions.status == STARTED:
            if(current_frame[0] == True):
                if tThisFlipGlobal > instructions.tStartRefresh + ins_duration-frameTolerance:
                    instructions.tStop = t  
                    instructions.frameNStop = frameN 
                    win.timeOnFlip(instructions, 'tStopRefresh') 
                    instructions.setAutoDraw(False)
            else:
                if tThisFlipGlobal > instructions.tStartRefresh + frame_duration-frameTolerance:
                    instructions.tStop = t 
                    instructions.frameNStop = frameN 
                    win.timeOnFlip(instructions, 'tStopRefresh') 
                    instructions.setAutoDraw(False)

        if fixation_point.status == NOT_STARTED and tThisFlip >= ISI-frameTolerance:

            fixation_point.frameNStart = frameN  
            fixation_point.tStart = t 
            fixation_point.tStartRefresh = tThisFlipGlobal 
            win.timeOnFlip(fixation_point, 'tStartRefresh') 
            fixation_point.setAutoDraw(True)
        if fixation_point.status == STARTED:
            if tThisFlipGlobal > fixation_point.tStartRefresh + frame_duration-frameTolerance:
                fixation_point.tStop = t  
                fixation_point.frameNStop = frameN 
                win.timeOnFlip(fixation_point, 'tStopRefresh')  
                fixation_point.setAutoDraw(False)
        if(current_frame[0] == True):
            waitOnFlip = False
            if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:

                key_resp.frameNStart = frameN  
                key_resp.tStart = t  
                key_resp.tStartRefresh = tThisFlipGlobal  
                win.timeOnFlip(key_resp, 'tStartRefresh') 
                key_resp.status = STARTED
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset) 
            if key_resp.status == STARTED:
                if tThisFlipGlobal > key_resp.tStartRefresh + ins_duration-frameTolerance:
                    key_resp.tStop = t 
                    key_resp.frameNStop = frameN  
                    win.timeOnFlip(key_resp, 'tStopRefresh') 
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space', 't', 'f'], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = [key.name for key in _key_resp_allKeys] 
                    key_resp.rt = [key.rt for key in _key_resp_allKeys]
                    correct = False
                    if(key_resp.keys[0] == 't'):
                        if(current_frame[7] == 'true'):
                            correct = True
                    if(key_resp.keys[0] == 'f'):
                        if(current_frame[7] == 'false'):
                            correct = True
                    key_resp.rt = [key.rt for key in _key_resp_allKeys]
                    if(current_frame[7] == None):
                        key_logger.writerow([frame_num, key_resp.keys, key_resp.rt, (100000 * frame_num - routineTimer.getTime()), "N/A"])
                    else:
                        key_logger.writerow([frame_num, key_resp.keys, key_resp.rt, (100000 * frame_num - routineTimer.getTime()), str(correct)])
                    continueRoutine = False
        if(current_frame[0] != True):
            waitOnFlip = False
            if key_resp.status == NOT_STARTED and tThisFlip >= ISI-frameTolerance:
                key_resp.frameNStart = frameN
                key_resp.tStart = t 
                key_resp.tStartRefresh = tThisFlipGlobal  
                win.timeOnFlip(key_resp, 'tStartRefresh') 
                key_resp.status = STARTED
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset) 
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard') 
            if key_resp.status == STARTED:
                if tThisFlipGlobal > key_resp.tStartRefresh + frame_duration-frameTolerance:
                    key_resp.tStop = t 
                    key_resp.frameNStop = frameN 
                    win.timeOnFlip(key_resp, 'tStopRefresh') 
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space', 't', 'f'], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = [key.name for key in _key_resp_allKeys] 
                    correct = False
                    if(key_resp.keys[0] == 't'):
                        if(current_frame[7] == 'true'):
                            correct = True
                    if(key_resp.keys[0] == 'f'):
                        if(current_frame[7] == 'false'):
                            correct = True
                    key_resp.rt = [key.rt for key in _key_resp_allKeys]
                    if(current_frame[7] == None):
                        key_logger.writerow([frame_num, key_resp.keys, key_resp.rt, (100000 * frame_num - routineTimer.getTime()), "N/A"])
                    else:
                        key_logger.writerow([frame_num, key_resp.keys, key_resp.rt, (100000 * frame_num - routineTimer.getTime()), str(correct)])

                    continueRoutine = False
        
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()

        if not continueRoutine: 
            break
        continueRoutine = False
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break 

        if continueRoutine: 
            win.flip()
        

    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    epochs.addData('stimulus.started', stimulus.tStartRefresh)
    epochs.addData('stimulus.stopped', stimulus.tStopRefresh)
    epochs.addData('instructions.started', instructions.tStartRefresh)
    epochs.addData('instructions.stopped', instructions.tStopRefresh)
    epochs.addData('fixation_point.started', fixation_point.tStartRefresh)
    epochs.addData('fixation_point.stopped', fixation_point.tStopRefresh)
    
    thisExp.nextEntry()

win.flip()

thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()

thisExp.abort()
win.close()

core.quit()

