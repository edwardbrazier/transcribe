from assembly_transcribe import TranscriptionApp
import tkinterdnd2 as tkdnd

if __name__ == "__main__":
    print("Starting TranscriptionApp...")
    root = tkdnd.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
