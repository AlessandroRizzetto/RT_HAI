import subprocess
import tkinter as tk
from tkinter import messagebox, ttk
from subprocess import Popen, PIPE
import os
import json
#import time
#import psutil

# variabili globali
p = None
m = None
v = None
pythonCommand = "py"

def is_python_program_running(program_name):
    tasklist_output = subprocess.check_output(['tasklist']).decode('utf-8')
    return any(program_name.lower() in line.lower() for line in tasklist_output.split('\n'))


def setup_audio(fileName, parameterToChange, value):
    file_paths = {
        'audio_features.pipeline-config': '../audio/pipes/audio_features.pipeline-config',
        'audio_input.pipeline-config': '../audio/pipes/audio_input.pipeline-config',
        'vad_filter.pipeline-config': '../audio/pipes/vad_filter.pipeline-config',
        'global.pipeline-config': '../audio/pipes/global.pipeline-config'
    }
    
    file_path = file_paths.get(fileName)
    if file_path:
        with open(file_path, 'r') as file:
            old_configs = file.readlines()
        
        with open(file_path, 'w') as file:
            for i, line in enumerate(old_configs):
                if parameterToChange == line.split(" = ")[0]:
                    old_configs[i] = f'{parameterToChange} = {value}\n'
                    break
            file.writelines(old_configs)

def start_feedback_system():
    global v
    print("STARTING FEEDBACK SYSTEM")
    v = Popen([pythonCommand, '../audio/scripts/feedback_execution.py'])
    

def start_video_action(is_online, is_configuration, crouch, hands, body_direction, configuration_time, path_to_video):
    # print(path_to_video)
    global m
    if not is_configuration and not is_online: 
        print("OFFLINE ANALYSIS")
        if path_to_video != "": 
            print("Using video file: ", path_to_video)
            with open('../MediaPipe/scripts/body_config.json', 'r+') as f:
                configurationData = json.load(f)
                configurationData['video_path'] = path_to_video
                f.seek(0)
                json.dump(configurationData, f, indent=4)
                f.truncate()
            m = Popen([pythonCommand, '../Mediapipe/video_analysis.py' , 'false', 'false'], stdin=PIPE)
        else:
            print("Path to video is empty, using default video")
            m = Popen([pythonCommand, '../Mediapipe/video_analysis.py' , 'false', 'false'], stdin=PIPE)
    elif not is_configuration and is_online:
        print("ONLINE ANALYSIS")
        if configuration_time != "" and configuration_time != None:
            configuration_time = float(configuration_time)
            print("Configuration time: ", configuration_time)
            with open('../MediaPipe/scripts/body_config.json', 'r+') as f:
                configurationData = json.load(f)
                configurationData['configuration_time'] = configuration_time
                f.seek(0)
                json.dump(configurationData, f, indent=4)
                f.truncate()
            m = Popen([pythonCommand, '../Mediapipe/video_analysis.py' , 'true', 'true'], stdin=PIPE)
        else:
            print("Configuration time is empty, using default configuration time")
            m = Popen([pythonCommand, '../Mediapipe/video_analysis.py' , 'true', 'true'], stdin=PIPE)
    elif is_configuration:
        print("ONLINE ANALYSIS - CONFIGURATION")
            
        if configuration_time != "" and configuration_time != None:
            configuration_time = float(configuration_time)
            print("Configuration time: ", configuration_time)
            with open('../MediaPipe/scripts/body_config.json', 'r+') as f:
                configurationData = json.load(f)
                configurationData['configuration_time'] = configuration_time
                f.seek(0)
                json.dump(configurationData, f, indent=4)
                f.truncate()
        if crouch == 1 and hands == 0 and body_direction == 0:
            m = Popen([pythonCommand, '../Mediapipe/scripts/bodyCalibrations.py', "true", "false", "false"], stdin=PIPE)
        elif hands == 1 and crouch == 0 and body_direction == 0:
            m = Popen([pythonCommand, '../Mediapipe/scripts/bodyCalibrations.py', "false", "true", "false"], stdin=PIPE)
        elif body_direction == 1 and crouch == 0 and hands == 0:
            m = Popen([pythonCommand, '../Mediapipe/scripts/bodyCalibrations.py', "false", "false", "true"], stdin=PIPE)
        elif crouch == 1 and hands == 1 and body_direction == 0:
            m = Popen([pythonCommand, '../Mediapipe/scripts/bodyCalibrations.py', "true", "true", "false"], stdin=PIPE)
        elif crouch == 1 and body_direction == 1 and hands == 0:
            m = Popen([pythonCommand, '../Mediapipe/scripts/bodyCalibrations.py', "true", "false", "true"], stdin=PIPE)
        elif hands == 1 and body_direction == 1 and crouch == 0:
            m = Popen([pythonCommand, '../Mediapipe/scripts/bodyCalibrations.py', "false", "true", "true"], stdin=PIPE)
        elif crouch == 1 and hands == 1 and body_direction == 1:
            m = Popen([pythonCommand, '../Mediapipe/scripts/bodyCalibrations.py', "true", "true", "true"], stdin=PIPE)
        else:
            # errore nessuna opzione selezionata
            print("Nessuna opzione selezionata")

def start_audio_action(audio_path, is_online, audioLive, audioLiveMic, vadCalibration, userCalibration, userCalibrationFile):
    global p
    global v
    if not is_online:
        print("OFFLINE ANALYSIS")
        setup_audio('global.pipeline-config', 'audio:live:mic', 'false')

        if not audioLive:
            setup_audio('global.pipeline-config', 'audio:live', 'false')
            if audio_path != "":
                # check if audio path is wav file
                if audio_path.endswith('.wav'):
                    
                    setup_audio('audio_input.pipeline-config', 'audio:file', audio_path)
                setup_audio('audio_features.pipeline-config', 'output:file:save', 'true')
            setup_audio('audio_features.pipeline-config', 'output:feedback:visual', 'false')
            setup_audio('audio_features.pipeline-config', 'output:feedback:aptic', 'false')
            print("Using audio file: ", audio_path)
        if vadCalibration:
            print("VAD Calibration")
            setup_audio('audio_features.pipeline-config', 'calibration:user:file:new', 'false')
            setup_audio('vad_filter.pipeline-config', 'vad:tresh:calibration', 'true')
            p = Popen([pythonCommand, '../audio/scripts/treshold_calibration.py'])
            v = Popen(['xmlpipe.exe', '-config', 'global', '../audio/pipes/vad_calibration.pipeline'])
        elif not vadCalibration:
            # Avvio analisi audio
            print("Audio Analysis")
            setup_audio('audio_features.pipeline-config', 'output:file:save', 'true') 
            setup_audio('audio_features.pipeline-config', 'output:file:new', 'true') 
            # v = Popen([pythonCommand, '../audio/scripts/feedback_execution.py'])
            p = Popen(['xmlpipe.exe', '-config', 'global', '../audio/pipes/audio_features.pipeline'])
        
    if is_online:
        print("ONLINE ANALYSIS")
        setup_audio('global.pipeline-config', 'audio:live', 'true')
        setup_audio('audio_input.pipeline-config', 'audio:live:mic', 'true')
        setup_audio('audio_features.pipeline-config', 'output:file:save', 'true')
        setup_audio('audio_features.pipeline-config', 'output:file:new', 'true')
        
        if vadCalibration:
            print("VAD Calibration")
            setup_audio('audio_features.pipeline-config', 'calibration:user', 'false')
            setup_audio('audio_features.pipeline-config', 'calibration:user:file:new', 'false')
            setup_audio('vad_filter.pipeline-config', 'vad:tresh:calibration', 'true')
            p = Popen([pythonCommand, '../audio/scripts/treshold_calibration.py'])
            v = Popen(['xmlpipe.exe', '-config', 'global', '../audio/pipes/vad_calibration.pipeline'])
        elif userCalibration == 1 and not vadCalibration:
            print("USER Calibration and Audio Analysis")
            setup_audio('vad_filter.pipeline-config', 'vad:tresh:calibration', 'false')
            setup_audio('audio_features.pipeline-config', 'calibration:user', 'true')
            setup_audio('audio_features.pipeline-config', 'calibration:user:file:new', 'true')
            # v = Popen([pythonCommand, '../audio/scripts/feedback_execution.py'])
            p = Popen(['xmlpipe.exe', '-config', 'global', '../audio/pipes/audio_features.pipeline'])
        elif userCalibration == 0 and not vadCalibration:
            print("Audio Analysis")
            setup_audio('vad_filter.pipeline-config', 'vad:tresh:calibration', 'false')
            setup_audio('audio_features.pipeline-config', 'calibration:user', 'false')
            setup_audio('audio_features.pipeline-config', 'calibration:user:file:new', 'false')
            setup_audio('audio_features.pipeline-config', 'output:feedback:visual', 'true')
            setup_audio('audio_features.pipeline-config', 'output:feedback:aptic', 'true')
            # v = Popen([pythonCommand, '../audio/scripts/feedback_execution.py'])
            p = Popen(['xmlpipe.exe', '-config', 'global', '../audio/pipes/audio_features.pipeline'])
            
            


try:
    os.chdir('./bin/')
    root = tk.Tk()
    root.title("HUMAN-AGENT INTERACTION SYSTEM - GUI")
    # root.geometry("1000x1000")

    # Inserisci immagine logo
    logo = tk.PhotoImage(file="../HAI_logo.png")
    # resize the image
    logo = logo.subsample(2, 2)
    logo_label = tk.Label(root, image=logo)
    logo_label.pack(pady=10)
    
    top_separator = ttk.Separator(root, orient='horizontal')
    top_separator.pack(fill='x', padx=10, pady=(0, 10))
    subtitle_label = tk.Label(root, text="FEEDBACK SYSTEM", font=("Helvetica", 14))
    subtitle_label.pack(pady=10)
    start_feedback_system_button = tk.Button(root, text="Start Feedback System", font=("Helvetica", 12), command=lambda: start_feedback_system())
    start_feedback_system_button.pack( pady=5)
    
    # start_video_button = tk.Button(left_frame, text="Start Video Analysis", command=lambda: start_video_action(False, False, None, None, None, None, video_entry.get()), font=("Helvetica", 12))


    # Sottotitolo "OFFLINE ANALYSIS"
    separator = ttk.Separator(root, orient='horizontal')
    separator.pack(fill='x', padx=10, pady=(0, 10))
    subtitle_label = tk.Label(root, text="OFFLINE ANALYSIS", font=("Helvetica", 14))
    subtitle_label.pack(pady=10)
    separator = ttk.Separator(root, orient='horizontal')
    separator.pack(fill='x', padx=10, pady=(10, 20))

    # Divide la finestra in due celle
    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)

    # Cella sinistra
    left_frame = tk.Frame(frame)
    left_frame.pack(side='left', fill='both', expand=True, padx=(5, 5))
    # Sottotitolo "VIDEO"
    left_subtitle = tk.Label(left_frame, text="VIDEO", font=("Helvetica", 12))
    left_subtitle.pack(pady=10)
    # Etichetta e campo per il percorso del video
    path_frame_video = tk.Frame(left_frame)
    path_frame_video.pack(pady=(0, 10), fill='x')
    path_to_video_label = tk.Label(path_frame_video, text="Path to video:", font=("Helvetica", 12))
    path_to_video_label.pack(side='top', pady=(0, 5))
    video_entry = tk.Entry(path_frame_video, font=("Helvetica", 12))
    video_entry.pack(side='right', fill='x', expand=True)

    # Pulsante per avviare l'analisi del video
    start_video_button = tk.Button(left_frame, text="Start Video Analysis", command=lambda: start_video_action(False, False, None, None, None, None, video_entry.get()), font=("Helvetica", 12))
    start_video_button.pack(side='bottom', pady=10)

    # Aggiungiamo la riga verticale tra le celle
    separator_vertical = ttk.Separator(frame, orient='vertical')
    separator_vertical.pack(side='left', fill='y', padx=5)

    # Cella destra
    right_frame = tk.Frame(frame)
    right_frame.pack(side='right', fill='both', expand=True, padx=(5, 5))
    # Sottotitolo "AUDIO"
    right_subtitle = tk.Label(right_frame, text="AUDIO", font=("Helvetica", 12))
    right_subtitle.pack(pady=10)
    # Etichetta e campo per il percorso dell'audio
    path_frame_audio = tk.Frame(right_frame)
    path_frame_audio.pack(pady=(0, 10), fill='x')
    path_to_audio_label = tk.Label(path_frame_audio, text="Path to audio:", font=("Helvetica", 12))
    path_to_audio_label.pack(side='top', pady=(0, 5))
    audio_entry = tk.Entry(path_frame_audio, font=("Helvetica", 12))
    audio_entry.pack(side='right', fill='x', expand=True)
    start_VAD_button = tk.Button(right_frame, text="Start VAD configuration", command=lambda: start_audio_action(audio_entry.get(), is_online=False, audioLive=False, audioLiveMic=False, vadCalibration=True, userCalibration=False, userCalibrationFile=""), font=("Helvetica", 12))
    start_VAD_button.pack(pady=10)
    
    start_audio_button = tk.Button(right_frame, text="Start Audio Analysis", command=lambda: start_audio_action(audio_entry.get(), is_online=False, audioLive=False, audioLiveMic=False, vadCalibration=False, userCalibration=False, userCalibrationFile=""), font=("Helvetica", 12))
    start_audio_button.pack(side='bottom', pady=10)

    #######################################################################################################################################################

    # Sottotitolo "ONLINE ANALYSIS"
    separator = ttk.Separator(root, orient='horizontal')
    separator.pack(fill='x', padx=10, pady=(0, 10))
    subtitle_label = tk.Label(root, text="ONLINE ANALYSIS", font=("Helvetica", 14))
    subtitle_label.pack(pady=10)
    separator = ttk.Separator(root, orient='horizontal')
    separator.pack(fill='x', padx=10, pady=(10, 20))

    # Divide la finestra in due celle
    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)

    # Cella sinistra
    left_frame = tk.Frame(frame)
    left_frame.pack(side='left', fill='both', expand=True, padx=(10, 5))
    left_subtitle = tk.Label(left_frame, text="VIDEO", font=("Helvetica", 12))
    left_subtitle.pack(pady=10)
    crouch_var = tk.IntVar()
    crouch_check = tk.Checkbutton(left_frame, text="Calibrate Crouch", font=("Helvetica", 12), variable=crouch_var)
    crouch_check.pack()
    hands_analysys_var = tk.IntVar()
    hands_analysys_check = tk.Checkbutton(left_frame, text="Calibrate Hands Distance", font=("Helvetica", 12), variable=hands_analysys_var)
    hands_analysys_check.pack()
    body_direction_var = tk.IntVar()
    body_direction_check = tk.Checkbutton(left_frame, text="Calibrate Body Direction", font=("Helvetica", 12), variable=body_direction_var)
    body_direction_check.pack()
    configuration_time_label = tk.Label(left_frame, text="Configuration time (s):", font=("Helvetica", 12))
    configuration_time_label.pack(pady=(10, 0))
    configuration_time_entry = tk.Entry(left_frame, font=("Helvetica", 12))
    configuration_time_entry.pack(pady=5)
    start_configuration_button = tk.Button(left_frame, text="Start Body Configuration", command=lambda: start_video_action(True, True, crouch_var.get(), hands_analysys_var.get(), body_direction_var.get(), configuration_time_entry.get(), None), font=("Helvetica", 12))
    start_configuration_button.pack(pady=10)
    start_video_button = tk.Button(left_frame, text="Start Video Analysis", command=lambda: start_video_action(True, False, crouch_var.get(), hands_analysys_var.get(), body_direction_var.get(), configuration_time_entry.get(), None), font=("Helvetica", 12))
    start_video_button.pack(pady=10)

    # Aggiungiamo la riga verticale tra le celle
    separator_vertical = ttk.Separator(frame, orient='vertical')
    separator_vertical.pack(side='left', fill='y', padx=5)

    # Cella destra
    right_frame = tk.Frame(frame)
    right_frame.pack(side='right', fill='both', expand=True, padx=(5, 10))
    right_subtitle = tk.Label(right_frame, text="AUDIO", font=("Helvetica", 12))
    right_subtitle.pack(pady=10)
    # vad_check = tk.Checkbutton(right_frame, text="VAD", font=("Helvetica", 12))
    # vad_check.pack()
    
    
    start_VAD_button = tk.Button(right_frame, text="Start VAD configuration", command=lambda: start_audio_action(None, is_online=True, audioLive=True, audioLiveMic=False, vadCalibration=True, userCalibration=False, userCalibrationFile=""), font=("Helvetica", 12))
    start_VAD_button.pack(pady=10)
    mini_separator_orizzontal = ttk.Separator(right_frame, orient='horizontal')
    mini_separator_orizzontal.pack(fill='x', pady=10)
    user_check_var = tk.IntVar()
    user_check = tk.Checkbutton(right_frame, text="Calibrate user audio", font=("Helvetica", 12), variable=user_check_var, pady=25)
    user_check.pack()
    
    start_audio_button = tk.Button(right_frame, text="Start Audio Analysis", command=lambda: start_audio_action(None, is_online=True, audioLive=True, audioLiveMic=False, vadCalibration=False, userCalibration=user_check_var.get(), userCalibrationFile=""), font=("Helvetica", 12))
    start_audio_button.pack(pady=10)

    root.mainloop()

except Exception as e:
    messagebox.showerror("Errore", f"Si è verificato un errore: {e}")

# def stop_audio(p):
#     p.communicate(input='\n'.encode())

# def stop_video(m):
#     print("Stop")
#     m.stdin.write('\n'.encode())

# def stop_analysis(m, p):
#     print("Stop analysis")
#     if m is not None:
#         print("Stop video")
#         m.terminate()
#     if p is not None:
#         p.communicate(input='\n'.encode())