import tkinter as tk
from tkinter import messagebox, ttk
from subprocess import Popen, PIPE
import os

# variabili globali
p = None
m = None

def start_video_action():
    print("Start")
    global m
    m = Popen(['python3', '../Mediapipe/video_analysis.py', 'false', 'true', '5.0'], stdin=PIPE)
    return m

def start_audio_action():
    print("Start audio")
    global p
    p = Popen(['xmlpipe.exe', '-config', 'global', '../audio/pipes/main.pipeline'], stdin=PIPE)
    return p
    
def stop_audio(p):
    p.communicate(input='\n'.encode())
    
def stop_video(m):
    print("Stop")
    m.stdin.write('\n'.encode())
    # m.terminate()
    
def start_analysis(check1, check2):
    print("Start analysis")
    global m
    global p
    check1 = check1.get()
    check2 = check2.get()
    if (check1 == 1 and check2 == 1 and m is None and p is None):
        command = ['xmlpipe.exe', '-config', 'global', '../audio/pipes/audio_features.pipeline']
        p = Popen(command, stdin=PIPE)
        m = start_video_action()
    if (check1 == 1 and check2 == 0) or (check1 == 1 and check2 == 1 and p is not None):
        m = start_video_action()
    if (check1 == 0 and check2 == 1 or (check1 == 1 and check2 == 1 and m is not None)):
        command = ['xmlpipe.exe', '-config', 'global', '../audio/pipes/audio_features.pipeline']
        p = Popen(command, stdin=PIPE)
    return m, p
    
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
    root.geometry("720x500")

    # Mostra le informazioni del programma
    # info_label = tk.Label(root, text="RT-HAI PROJECT", font=("Helvetica", 16))
    # info_label.pack(pady=10)
    
    # Inserisci immagine logo
    logo = tk.PhotoImage(file="../HAI_logo.png")
    # resize the image
    logo = logo.subsample(2, 2)
    logo_label = tk.Label(root, image=logo)
    logo_label.pack(pady=10)
    

    # Crea un'etichetta e un campo di testo per l'input
    # input_label = tk.Label(root, text="Inserisci qualcosa:", font=("Helvetica", 12))
    # input_label.pack()
    # input_entry = tk.Entry(root, font=("Helvetica", 12))
    # input_entry.pack(pady=5) 

    # Aggiungi un controllo slider
    # slider_label = tk.Label(root, text="Seleziona un valore:", font=("Helvetica", 12))
    # slider_label.pack()
    # slider = ttk.Scale(root, from_=0, to=100, orient="horizontal")
    # slider.pack(pady=5)    
    # slider.set(50)
    
    # Aggiungi un pulsante per ottenere il valore dello slider
    # def get_slider_value():
    #     messagebox.showinfo("Valore slider", f"Il valore selezionato è: {slider.get()}")
    # slider_button = tk.Button(root, text="Mostra valore slider", command=get_slider_value, font=("Helvetica", 12))
    # slider_button.pack(pady=5)
    
    # crea due checkbox
    check1 = tk.IntVar()
    check2 = tk.IntVar()
    check1.set(0)
    check2.set(0)
    check1_label = tk.Checkbutton(root, text="Video Analysis", variable=check1, font=("Helvetica", 12))
    check1_label.pack()
    check2_label = tk.Checkbutton(root, text="Audio Analysis", variable=check2, font=("Helvetica", 12))
    check2_label.pack()
    
    start_analysis_button = tk.Button(root, text="Start Analysis", command=lambda: start_analysis(check1, check2), font=("Helvetica", 12))
    start_analysis_button.pack(pady=10)
    stop_analysis_button = tk.Button(root, text="Stop Analysis", command=lambda: stop_analysis(m, p), font=("Helvetica", 12))
    stop_analysis_button.pack(pady=10)

    root.mainloop()

except Exception as e:
    messagebox.showerror("Errore", f"Si è verificato un errore: {e}")
