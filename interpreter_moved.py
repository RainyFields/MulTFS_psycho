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
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard
import pandas as pd
import random
from PIL import Image
import json

import argparse

root = os.getcwd()
print(root)
def LoadFromFile(namespace):
    with open(root + "/config.json") as f:
        data = json.load(f)
    return data[namespace]


parser = argparse.ArgumentParser()
# other arguments
parser.add_argument('--stim_path', type = str, required = False, default=LoadFromFile("stim_path"))
parser.add_argument('--task', type = str, required = False, default = LoadFromFile("task"))
parser.add_argument('--img_path', type = str, required = False, default = LoadFromFile("img_path"))
parser.add_argument('--n_trials', type = int, required = False, default = LoadFromFile("n_trials"))
args = parser.parse_args()

if args.task == "1-back_identity":
    task_dir = root + "/1back_identity"

#print(pd.__version__)
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.1.2'
expName = 'interpreter'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='./interpreter_moved.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=(500, 500), fullscr=False, screen=0,
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# Setup ioHub
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

filelist = []
rootdir = task_dir
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if(os.path.isdir(d)):
        filelist.append(d) 
filelist.pop()
filelist.remove(filelist[0])
#print(filelist)

random_index_list = []
for i in range(args.n_trials):
    rand_gen = random.randint(0, len(filelist)-1)
    while (rand_gen in random_index_list):
        rand_gen = random.randint(0, len(filelist)-1)
    random_index_list.append(rand_gen)
frame_log = {}
num_total_epochs = 0
current_frame_num = 0
for a in range(args.n_trials):
    task_number = str(filelist[a])
    folder_number = task_number
    task_path = os.path.join(f'{task_number}', 'compo_task_example')

    with open(task_path, "r") as fobj:
        content = json.load(fobj)
        
    trial_total_epochs = content['epochs']
    objects = content['objects']
    df = pd.read_pickle(root + args.stim_path)
    instr = content['instruction']
    answers = content['answers']
    img_log = {}
    #img_log[-1] = [True, None, None, None, None, None, None, instr] 
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
        img_path = os.path.join(root + args.img_path, f'{ref}/image.png')
        im = Image.open(img_path, 'r')
        img_log[img_epoch] = [is_instr, im, location, ctg_mod, obj_mod, ang_mod, is_distractor, curr_answer, a, None]
    for j in range(trial_total_epochs):
        if img_log.get(j) == None:
            img_log[j] = [False, None, None, None, None, None, None, None, a, None]
    for k in range(len(img_log)):
        frame_log[current_frame_num] = img_log.get(k)
        current_frame_num += 1
    num_total_epochs = num_total_epochs + trial_total_epochs + 1



# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# Initialize components for Routine "trial"
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


# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# set up handler to look after randomisation of conditions etc
epochs = data.TrialHandler(nReps=num_total_epochs + 1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='epochs')
thisExp.addLoop(epochs)  # add the loop to the experiment
thisEpoch = epochs.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisEpoch.rgb)
if thisEpoch != None:
    for paramName in thisEpoch:
        exec('{} = thisEpoch[paramName]'.format(paramName))
frame_num = 0
for thisEpoch in epochs:
    currentLoop = epochs
    # abbreviate parameter names if possible (e.g. rgb = thisEpoch.rgb)
    if thisEpoch != None:
        for paramName in thisEpoch:
            exec('{} = thisEpoch[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trial"-------
    continueRoutine = True
    routineTimer.add(100000.000000)
    # update component parameters for each repeat
    current_frame = frame_log.get(frame_num)
    if(current_frame[0] == None or current_frame[0] == False ):
        instructions.setText("")
        fixation_point.setSize((0.05, 0.05))
        if(current_frame[1] == None):
            stimulus.setImage(None)
        else:
            stimulus.setPos(current_frame[2])
            #stimulus.setPos((0, -0.5))
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
    # keep track of which components have finished
    trialComponents = [stimulus, instructions, fixation_point, key_resp]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = trialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *stimulus* updates
        if stimulus.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            stimulus.frameNStart = frameN  # exact frame index
            stimulus.tStart = t  # local t and not account for scr refresh
            stimulus.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimulus, 'tStartRefresh')  # time at next scr refresh
            stimulus.setAutoDraw(True)
        if stimulus.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > stimulus.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                stimulus.tStop = t  # not accounting for scr refresh
                stimulus.frameNStop = frameN  # exact frame index
                win.timeOnFlip(stimulus, 'tStopRefresh')  # time at next scr refresh
                stimulus.setAutoDraw(False)
        
        # *instructions* updates
        if instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions.frameNStart = frameN  # exact frame index
            instructions.tStart = t  # local t and not account for scr refresh
            instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions, 'tStartRefresh')  # time at next scr refresh
            instructions.setAutoDraw(True)
        if instructions.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if(current_frame[0] == True):
                if tThisFlipGlobal > instructions.tStartRefresh + 10000-frameTolerance:
                # keep track of stop time/frame for later
                    instructions.tStop = t  # not accounting for scr refresh
                    instructions.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(instructions, 'tStopRefresh')  # time at next scr refresh
                    instructions.setAutoDraw(False)
            else:
                if tThisFlipGlobal > instructions.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    instructions.tStop = t  # not accounting for scr refresh
                    instructions.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(instructions, 'tStopRefresh')  # time at next scr refresh
                    instructions.setAutoDraw(False)
        
        # *fixation_point* updates
        if fixation_point.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_point.frameNStart = frameN  # exact frame index
            fixation_point.tStart = t  # local t and not account for scr refresh
            fixation_point.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_point, 'tStartRefresh')  # time at next scr refresh
            fixation_point.setAutoDraw(True)
        if fixation_point.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation_point.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                fixation_point.tStop = t  # not accounting for scr refresh
                fixation_point.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fixation_point, 'tStopRefresh')  # time at next scr refresh
                fixation_point.setAutoDraw(False)
        if(current_frame[0] == True):
        # *key_resp* updates
            waitOnFlip = False
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 10000-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(key_resp, 'tStopRefresh')  # time at next scr refresh
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space'], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    # a response ends the routine
                    continueRoutine = False
        if(current_frame[0] != True):
            # *key_resp* updates
            waitOnFlip = False
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(key_resp, 'tStopRefresh')  # time at next scr refresh
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space'], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    # a response ends the routine
                    continueRoutine = False

        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial"-------
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

    
# completed epochs repeats of 'epochs'


# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
