import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import assemblyai as aai
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
import tkinterdnd2 as tkdnd
from typing import List, Optional
import xml.etree.ElementTree as ET

class TranscriptionApp:
    def __init__(self, root: tk.Tk) -> None:
        """Initializes the TranscriptionApp.

        Preconditions:
            The environment variable 'assemblyAI_key' must be set with a valid AssemblyAI API key.

        Args:
            root: The root Tkinter window.

        Raises:
            EnvironmentError: If the 'assemblyAI_key' environment variable is not set.
        """
        self.root = root
        self.root.title("Audio Transcriber")

        self.api_key = os.getenv("assemblyAI_key")
        if not self.api_key:
            raise EnvironmentError("API key for AssemblyAI is not set in environment variables.")

        aai.settings.api_key = self.api_key

        self.transcriber = aai.Transcriber()

        self.create_widgets()
        self.root.drop_target_register(tkdnd.DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_files)
        self.transcribed_files: List[Path] = []

    def create_widgets(self) -> None:
        """Creates and places the widgets in the Tkinter window."""
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.drop_area = tk.Listbox(self.frame, selectmode=tk.MULTIPLE, height=10)
        self.drop_area.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.progress = ttk.Progressbar(self.frame, mode='determinate')
        self.progress.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.transcribe_button = ttk.Button(self.frame, text="Transcribe", command=self.transcribe_files)
        self.transcribe_button.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.output_area = tk.Listbox(self.frame, height=10)
        self.output_area.grid(row=3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)

    def drop_files(self, event: tk.Event) -> None:
        """Handles file drop events and adds valid files to the drop_area Listbox.

        Args:
            event: The Tkinter event object.

        Side effects:
            Displays an error message if an unsupported file type is dropped.

        Raises:
            None
        """
        files = self.root.tk.splitlist(event.data)
        for file in files:
            if file.endswith(('.mp3', '.wav', '.mp4', '.m4a')):
                self.drop_area.insert(tk.END, file)
            else:
                messagebox.showerror("Invalid File Type", f"Unsupported file type: {file}")

    def transcribe_files(self) -> None:
        """Transcribes the files listed in the drop_area Listbox.

        Preconditions:
            Files must be added to the drop_area Listbox.

        Side effects:
            Updates the progress bar and the output_area Listbox with the transcription results.

        Raises:
            None
        """
        self.transcribed_files = []
        files = self.drop_area.get(0, tk.END)
        self.progress['value'] = 0
        self.progress['maximum'] = len(files)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.transcribe_file, file) for file in files]
            for future in futures:
                future.add_done_callback(self.update_progress)

    def transcribe_file(self, file_path: str) -> Path:
        """Transcribes a single file and writes the result to a text file.

        Args:
            file_path: The path to the file to be transcribed.

        Returns:
            The path to the output text file.

        Raises:
            Exception: If an error occurs during transcription.
        """
        file_path = Path(file_path)
        transcript = self.transcriber.transcribe(file_path.as_posix())
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(transcript.error)
        
        output_path = file_path.with_suffix(".txt")
        with open(output_path, 'w') as f:
            f.write(transcript.text)
        
        return output_path

    def update_progress(self, future: Future) -> None:
        """Updates the progress bar and adds the output file to the output_area Listbox.

        Args:
            future: The Future object representing the transcription task.

        Side effects:
            Updates the progress bar and the output_area Listbox.
            Displays an error message if an exception occurs.

        Raises:
            None
        """
        self.progress['value'] += 1
        try:
            output_file = future.result()
            self.output_area.insert(tk.END, output_file)
            self.transcribed_files.append(output_file)
            if self.progress['value'] == self.progress['maximum']:
                self.create_xml_summary()
        except Exception as e:
            messagebox.showerror("Transcription Error", str(e))

    def create_xml_summary(self) -> None:
        """Creates an XML file summarizing all transcribed files with their paths and contents.

        Side effects:
            Creates an XML file in the directory of the first transcribed file.

        Raises:
            None
        """
        if not self.transcribed_files:
            return

        root_element = ET.Element("Transcriptions")
        for file_path in self.transcribed_files:
            file_path = Path(file_path)
            with open(file_path, 'r') as f:
                transcript_text = f.read()
            
            transcription_element = ET.SubElement(root_element, "Transcription")
            ET.SubElement(transcription_element, "FilePath").text = str(file_path.with_suffix(file_path.suffix))
            ET.SubElement(transcription_element, "Content").text = transcript_text

        tree = ET.ElementTree(root_element)
        summary_path = Path(self.transcribed_files[0]).parent / "transcriptions_summary.xml"
        tree.write(summary_path, encoding="utf-8", xml_declaration=True)

        self.output_area.insert(tk.END, f"Summary XML created at {summary_path}")

if __name__ == "__main__":
    root = tkdnd.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
