import tkinter as tk
from tkinter import messagebox, ttk
from subprocess import Popen, PIPE
import os
import json

# variabili globali
p = None
m = None

def start_video_action(is_online, is_configuration, crouch, hands, body_direction, configuration_time, path_to_video):
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
            m = Popen(['python3', '../Mediapipe/video_analysis.py' , 'false', 'false'], stdin=PIPE)
        else:
            print("Path to video is empty, using default video")
            m = Popen(['python3', '../Mediapipe/video_analysis.py' , 'false', 'false'], stdin=PIPE)
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
            m = Popen(['python3', '../Mediapipe/video_analysis.py' , 'false', 'true'], stdin=PIPE)
        else:
            print("Configuration time is empty, using default configuration time")
            m = Popen(['python3', '../Mediapipe/video_analysis.py' , 'false', 'true'], stdin=PIPE)
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
            m = Popen(['python3', '../Mediapipe/scripts/bodyCalibrations.py', "true", "false", "false"], stdin=PIPE)
        elif hands == 1 and crouch == 0 and body_direction == 0:
            m = Popen(['python3', '../Mediapipe/scripts/bodyCalibrations.py', "false", "true", "false"], stdin=PIPE)
        elif body_direction == 1 and crouch == 0 and hands == 0:
            m = Popen(['python3', '../Mediapipe/scripts/bodyCalibrations.py', "false", "false", "true"], stdin=PIPE)
        elif crouch == 1 and hands == 1 and body_direction == 0:
            m = Popen(['python3', '../Mediapipe/scripts/bodyCalibrations.py', "true", "true", "false"], stdin=PIPE)
        elif crouch == 1 and body_direction == 1 and hands == 0:
            m = Popen(['python3', '../Mediapipe/scripts/bodyCalibrations.py', "true", "false", "true"], stdin=PIPE)
        elif hands == 1 and body_direction == 1 and crouch == 0:
            m = Popen(['python3', '../Mediapipe/scripts/bodyCalibrations.py', "false", "true", "true"], stdin=PIPE)
        elif crouch == 1 and hands == 1 and body_direction == 1:
            m = Popen(['python3', '../Mediapipe/scripts/bodyCalibrations.py', "true", "true", "true"], stdin=PIPE)
        else:
            # errore nessuna opzione selezionata
            print("Nessuna opzione selezionata")
        

def start_audio_action():
    print("Start audio")
    global p
    p = Popen(['xmlpipe.exe', '-config', 'global', '../audio/pipes/main.pipeline'], stdin=PIPE)

def stop_audio(p):
    p.communicate(input='\n'.encode())

def stop_video(m):
    print("Stop")
    m.stdin.write('\n'.encode())

def stop_analysis(m, p):
    print("Stop analysis")
    if m is not None:
        print("Stop video")
        m.terminate()
    if p is not None:
        p.communicate(input='\n'.encode())

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
    path_to_video_label.pack(side='left', padx=(0, 5))
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
    right_frame.pack(side='right', fill='both', expand=True, padx=(5, 10))
    right_subtitle = tk.Label(right_frame, text="AUDIO", font=("Helvetica", 12))
    right_subtitle.pack(pady=10)
    vad_check = tk.Checkbutton(right_frame, text="VAD", font=("Helvetica", 12))
    vad_check.pack()
    user_check = tk.Checkbutton(right_frame, text="Utente", font=("Helvetica", 12))
    user_check.pack()
    audio_entry = tk.Entry(right_frame, font=("Helvetica", 12))
    audio_entry.pack(pady=5)
    start_audio_button = tk.Button(right_frame, text="Start Audio Analysis", command=start_audio_action, font=("Helvetica", 12))
    start_audio_button.pack(pady=10)

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
    crouch_check = tk.Checkbutton(left_frame, text="Crouch", font=("Helvetica", 12), variable=crouch_var)
    crouch_check.pack()
    hands_analysys_var = tk.IntVar()
    hands_analysys_check = tk.Checkbutton(left_frame, text="Hands", font=("Helvetica", 12), variable=hands_analysys_var)
    hands_analysys_check.pack()
    body_direction_var = tk.IntVar()
    body_direction_check = tk.Checkbutton(left_frame, text="Body Direction", font=("Helvetica", 12), variable=body_direction_var)
    body_direction_check.pack()
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
    vad_check = tk.Checkbutton(right_frame, text="VAD", font=("Helvetica", 12))
    vad_check.pack()
    user_check = tk.Checkbutton(right_frame, text="Utente", font=("Helvetica", 12))
    user_check.pack()
    audio_entry = tk.Entry(right_frame, font=("Helvetica", 12))
    audio_entry.pack(pady=5)
    start_VAD_button = tk.Button(right_frame, text="Start VAD configuration", command=start_audio_action, font=("Helvetica", 12))
    start_VAD_button.pack(pady=10)
    start_audio_button = tk.Button(right_frame, text="Start Audio Analysis", command=start_audio_action, font=("Helvetica", 12))
    start_audio_button.pack(pady=10)

    root.mainloop()

except Exception as e:
    messagebox.showerror("Errore", f"Si è verificato un errore: {e}")
