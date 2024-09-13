import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import assemblyai as aai
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
import tkinterdnd2 as tkdnd
from typing import List, Optional
import xml.etree.ElementTree as ET
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np  # Required for handling audio data
import time
import openai  # For interacting with the OpenAI API

class TranscriptionApp:
    def __init__(self, root: tk.Tk) -> None:
        """Initializes the TranscriptionApp with two panels and AI integration.

        Left Panel:
            - Button to record and stop audio recording.
            - Pause/Resume button to pause and resume recording.
            - Automatic transcription upon stopping.
            - Transcript displayed in a new window with a "Tidy Up" button.

        Right Panel:
            - Existing functionality to drop audio files for transcription.

        AI Integration:
            - Uses OpenAI's GPT-4o-mini model to tidy up punctuation and capitalization.

        Preconditions:
            The environment variable 'assemblyAI_key' must be set with a valid API key for both AssemblyAI and OpenAI.

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

        # Set up API keys for AssemblyAI and OpenAI
        aai.settings.api_key = self.api_key
        openai.api_key = self.api_key

        self.transcriber = aai.Transcriber()

        # Initialize recording attributes
        self.is_recording = False
        self.is_paused = False
        self.recording_thread = None
        self.audio_data = []
        self.samplerate = 44100  # Standard sampling rate for audio

        self.create_widgets()
        self.root.drop_target_register(tkdnd.DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_files)
        self.transcribed_files: List[Path] = []

        print("TranscriptionApp initialized.")

    def create_widgets(self) -> None:
        """Creates and places the widgets in the Tkinter window, including two panels."""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left Panel Frame
        self.left_frame = ttk.Frame(self.main_frame, padding="10")
        self.left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Right Panel Frame
        self.right_frame = ttk.Frame(self.main_frame, padding="10")
        self.right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        # Left Panel Widgets
        self.record_button = ttk.Button(self.left_frame, text="Record", command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))

        self.pause_button = ttk.Button(self.left_frame, text="Pause", command=self.toggle_pause, state='disabled')
        self.pause_button.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))

        # Right Panel Widgets (Existing functionality)
        self.drop_area_label = ttk.Label(self.right_frame, text="Files to Transcribe")
        self.drop_area_label.grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=(tk.W, tk.E))

        self.drop_area = tk.Listbox(self.right_frame, selectmode=tk.MULTIPLE, height=10)
        self.drop_area.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.clear_button = ttk.Button(self.right_frame, text="Clear Files", command=self.clear_files)
        self.clear_button.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))

        self.transcribe_button = ttk.Button(self.right_frame, text="Transcribe", command=self.transcribe_files)
        self.transcribe_button.grid(row=2, column=1, pady=5, sticky=(tk.W, tk.E))

        self.progress = ttk.Progressbar(self.right_frame, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.output_area_label = ttk.Label(self.right_frame, text="Transcription Results")
        self.output_area_label.grid(row=4, column=0, columnspan=2, pady=(5, 0), sticky=(tk.W, tk.E))

        self.output_area = tk.Listbox(self.right_frame, height=10)
        self.output_area.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.right_frame.rowconfigure(1, weight=1)
        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.columnconfigure(1, weight=1)

        print("Widgets created.")

    def toggle_recording(self) -> None:
        """Toggles the recording state between start and stop."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> None:
        """Starts recording audio from the microphone."""
        self.is_recording = True
        self.is_paused = False
        self.record_button.config(text="Stop")
        self.pause_button.config(state='normal', text="Pause")
        self.audio_data = []  # Reset audio data
        self.stop_recording_event = threading.Event()
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()
        print("Recording started.")

    def stop_recording(self) -> None:
        """Stops recording audio and initiates transcription."""
        self.is_recording = False
        self.is_paused = False
        self.record_button.config(text="Record")
        self.pause_button.config(state='disabled', text="Pause")
        self.stop_recording_event.set()  # Signal the recording thread to stop
        self.recording_thread.join()  # Wait for the recording thread to finish
        print("Recording stopped.")
        self.save_and_transcribe_recording()

    def toggle_pause(self) -> None:
        """Toggles the pause state between pause and resume."""
        if not self.is_paused:
            self.is_paused = True
            self.pause_button.config(text="Resume")
            print("Recording paused.")
        else:
            self.is_paused = False
            self.pause_button.config(text="Pause")
            print("Recording resumed.")

    def record_audio(self) -> None:
        """Records audio in a separate thread and stores the data."""
        def callback(indata, frames, time, status):
            """Callback function to collect audio data."""
            if not self.is_paused:
                self.audio_data.append(indata.copy())

        with sd.InputStream(samplerate=self.samplerate, channels=1, callback=callback):
            while not self.stop_recording_event.is_set():
                time.sleep(0.1)  # Sleep to prevent high CPU usage

        print("Recording thread finished.")

    def save_and_transcribe_recording(self) -> None:
        """Saves the recorded audio to a file and transcribes it."""
        if not self.audio_data:
            messagebox.showwarning("Recording Error", "No audio data recorded.")
            return

        # Concatenate all recorded audio data
        audio_data = np.concatenate(self.audio_data, axis=0)

        # Save the audio data to a WAV file
        recorded_file_path = Path("recorded_audio.wav")
        sf.write(recorded_file_path, audio_data, self.samplerate)
        print(f"Audio saved to {recorded_file_path}")

        # Transcribe the recorded audio
        threading.Thread(target=self.transcribe_recorded_file, args=(recorded_file_path,)).start()

    def transcribe_recorded_file(self, file_path: Path) -> None:
        """Transcribes the recorded audio file and displays the transcript.

        Args:
            file_path: The path to the recorded audio file.
        """
        try:
            print(f"Transcribing recorded file: {file_path}")
            transcript = self.transcriber.transcribe(file_path.as_posix())
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(transcript.error)

            # Display the transcript in a new window
            self.display_transcript(transcript.text)
            print(f"Transcription completed for recorded file: {file_path}")
        except Exception as e:
            messagebox.showerror("Transcription Error", str(e))

    def display_transcript(self, text: str) -> None:
        """Displays the transcript in a new window with a "Tidy Up" button.

        Args:
            text: The transcribed text to display.
        """
        transcript_window = tk.Toplevel(self.root)
        transcript_window.title("Transcript")

        text_widget = tk.Text(transcript_window, wrap='word')
        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.NORMAL)  # Make editable for updates
        text_widget.pack(expand=True, fill='both')

        # Add "Tidy Up" button
        tidy_button = ttk.Button(
            transcript_window,
            text="Tidy Up",
            command=lambda: self.tidy_transcript(text_widget)
        )
        tidy_button.pack(pady=5)

    def tidy_transcript(self, text_widget: tk.Text) -> None:
        """Tidies up the transcript by fixing punctuation and capitalization using AI.

        Args:
            text_widget: The Text widget containing the transcript.

        Side effects:
            - Updates the text_widget with the tidied transcript.
            - Displays an error message if the AI call fails.
        """
        # Get the current text from the text widget
        text = text_widget.get("1.0", tk.END).strip()

        # Define the system prompt for the AI
        system_prompt = (
            "You are an assistant that tidies up transcripts by fixing punctuation and capitalization, "
            "without changing the wording or meaning. Please correct the punctuation and capitalization "
            "in the text provided, without changing anything more than necessary."
        )

        # Prepare the messages for the AI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=4000,
                temperature=0
            )

            # Get the assistant's reply
            reply = response['choices'][0]['message']['content']

            # Update the text widget with the tidied-up text
            text_widget.config(state=tk.NORMAL)
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, reply)
            text_widget.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def clear_files(self) -> None:
        """Clears the list of files to transcribe."""
        self.drop_area.delete(0, tk.END)
        print("Files cleared.")

    def drop_files(self, event: tk.Event) -> None:
        """Handles file drop events and adds valid files to the drop_area Listbox.

        Args:
            event: The Tkinter event object.

        Side effects:
            Displays an error message if an unsupported file type is dropped.
        """
        files = self.root.tk.splitlist(event.data)
        for file in files:
            if file.endswith(('.mp3', '.wav', '.mp4', '.m4a')):
                self.drop_area.insert(tk.END, file)
                print(f"File added: {file}")
            else:
                messagebox.showerror("Invalid File Type", f"Unsupported file type: {file}")

    def transcribe_files(self) -> None:
        """Transcribes the files listed in the drop_area Listbox.

        Side effects:
            Updates the progress bar and the output_area Listbox with the transcription results.
        """
        self.transcribed_files = []
        files = self.drop_area.get(0, tk.END)
        if not files:
            messagebox.showwarning("No Files", "Please add files to transcribe.")
            return

        self.progress['value'] = 0
        self.progress['maximum'] = len(files)

        print("Transcription started.")

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
        """
        file_path = Path(file_path)
        print(f"Transcribing file: {file_path}")
        transcript = self.transcriber.transcribe(file_path.as_posix())
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(transcript.error)

        output_path = file_path.with_suffix(".txt")
        with open(output_path, 'w') as f:
            f.write(transcript.text)

        self.transcribed_files.append(output_path)
        print(f"Transcription completed for file: {file_path}")
        return output_path

    def update_progress(self, future: Future) -> None:
        """Updates the progress bar and adds a 'done' message to the output_area Listbox.

        Args:
            future: The Future object representing the transcription task.

        Side effects:
            Updates the progress bar and the output_area Listbox.
            Displays an error message if an exception occurs.
        """
        try:
            result = future.result()
            self.progress['value'] += 1
            self.output_area.insert(tk.END, f"Transcription done: {result}")
            print(f"Progress updated. Transcription done: {result}")
            if self.progress['value'] == self.progress['maximum']:
                self.create_xml_summary()
        except Exception as e:
            messagebox.showerror("Transcription Error", str(e))

    def create_xml_summary(self) -> None:
        """Creates an XML file summarizing all transcribed files with their paths and contents.

        Side effects:
            Creates an XML file in the directory of the first transcribed file.
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
        print(f"XML summary created at: {summary_path}")

if __name__ == "__main__":
    root = tkdnd.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
