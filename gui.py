import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
from subprocess import Popen, PIPE
import os
import signal

# variabili globali
p = None
m = None

def start_video_action():
    print("Start")
    # subprocess.run('dir', shell=True)
    # subprocess.run('dir', shell=True)
    global m
    m = Popen(['powershell', 'python3 ..\Mediapipe\MediaPipe.py'], shell=True, stdin=PIPE)
    # subprocess.run(['powershell', 'python3 ..\Mediapipe\MediaPipe.py'], shell=True)
    return m

def start_audio_action():
    print("Start audio")
    # subprocess.run('dir', shell=True)
    # subprocess.run('dir', shell=True)
    global p
    p = Popen(['powershell', 'xmlpipe.exe -config global ..\audio\pipes\main.pipeline'], shell=True, stdin=PIPE)
    return p
    
def stop_audio(p):
    # messagebox.showinfo("Stop", "Hai premuto il pulsante Stop!")
    p.communicate(input='\n'.encode())
    
def stop_video(m):
    print("Stop")
    m.terminate()
    
def start_analysis(check1, check2):
    print("Start analysis")
    global m
    global p
    check1 = check1.get()
    check2 = check2.get()
    if (check1 == 1 and check2 == 1 and m is None and p is None):
        p = Popen(['powershell', 'xmlpipe.exe -config global ..\audio\pipes\main.pipeline'], shell=True, stdin=PIPE)
        m = Popen(['powershell', 'python3 ..\Mediapipe\MediaPipe.py'], shell=True, stdin=PIPE)
    if (check1 == 1 and check2 == 0) or (check1 == 1 and check2 == 1 and p is not None):
        m = Popen(['powershell', 'python3 ..\Mediapipe\MediaPipe.py'], shell=True, stdin=PIPE)
    if (check1 == 0 and check2 == 1 or (check1 == 1 and check2 == 1 and m is not None)):
        p = Popen(['powershell', 'xmlpipe.exe -config global ..\audio\pipes\main.pipeline'], shell=True, stdin=PIPE)
    return m, p
    
def stop_analysis(m, p):
    print("Stop analysis")
    if m is not None:
        print("Stop video")
        m.terminate()  # Prova a terminare il processo in modo pulito
        # Se il processo non termina, prova a killarlo
        try:
            m.kill()
        except Exception as e:
            print(f"Errore durante la terminazione del processo video: {e}")
    if p is not None:
        p.communicate(input='\n'.encode())  # Termina il processo audio normalmente
    
    

try:
    os.chdir('./bin/')
    root = tk.Tk()
    root.title("RT-HAI PROJECT - GUI")
    root.geometry("600x400")  # Imposta le dimensioni della finestra

    # Mostra le informazioni del programma
    info_label = tk.Label(root, text="RT-HAI PROJECT", font=("Helvetica", 16))
    info_label.pack(pady=10)

    # Crea un'etichetta e un campo di testo per l'input
    input_label = tk.Label(root, text="Inserisci qualcosa:", font=("Helvetica", 12))
    input_label.pack()
    input_entry = tk.Entry(root, font=("Helvetica", 12))
    input_entry.pack(pady=5) 
    # salva il valore dell'input
    input_value = input_entry.get()

    # Aggiungi un controllo slider
    slider_label = tk.Label(root, text="Seleziona un valore:", font=("Helvetica", 12))
    slider_label.pack()
    slider = ttk.Scale(root, from_=0, to=100, orient="horizontal")
    slider.pack(pady=5)    
    slider.set(50)
    
    # Aggiungi un pulsante per ottenere il valore dello slider
    def get_slider_value():
        messagebox.showinfo("Valore slider", f"Il valore selezionato è: {slider.get()}")
    slider_button = tk.Button(root, text="Mostra valore slider", command=get_slider_value, font=("Helvetica", 12))
    slider_button.pack(pady=5)
    # salva il valore dello slider
    slider_value = slider.get()
    
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
