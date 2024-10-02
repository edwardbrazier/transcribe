import os
import tkinter as tk
from tkinter import ttk, messagebox
import assemblyai as aai
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
import tkinterdnd2 as tkdnd
from typing import List
import xml.etree.ElementTree as ET
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import openai
import difflib

class TranscriptionApp:
    def __init__(self, root: tk.Tk) -> None:
        """Initializes the TranscriptionApp with recording, transcription, and AI integration."""
        self.root = root
        self.root.title("Audio Transcriber")

        # Get API keys
        self.aai_api_key = os.getenv("assemblyAI_key")
        if not self.aai_api_key:
            raise EnvironmentError("AssemblyAI API key is not set.")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise EnvironmentError("OpenAI API key is not set.")

        # Set up API keys
        aai.settings.api_key = self.aai_api_key
        openai.api_key = self.openai_api_key

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI()

        self.transcriber = aai.Transcriber()

        # Recording attributes
        self.is_recording = False
        self.is_paused = False
        self.recording_thread = None
        self.audio_data = []
        self.samplerate = 44100

        # Store previous transcript version
        self.previous_transcript = ""

        # Initialize lock for thread-safe operations
        self.transcribed_files: List[Path] = []
        self.transcribed_files_lock = threading.Lock()  # Add this line

        self.create_widgets()
        self.root.drop_target_register(tkdnd.DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_files)

        # Add keyboard shortcut for recording
        self.root.bind('<Shift-R>', self.toggle_recording)

        print("TranscriptionApp initialized.")


    def create_widgets(self) -> None:
        """Creates and places the widgets in the Tkinter window."""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=tk.NSEW)

        # Left Panel Frame
        self.left_frame = ttk.Frame(self.main_frame, padding="10")
        self.left_frame.grid(row=0, column=0, sticky=tk.NSEW)

        # Right Panel Frame
        self.right_frame = ttk.Frame(self.main_frame, padding="10")
        self.right_frame.grid(row=0, column=1, sticky=tk.NSEW)

        # Configure weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        # Left Panel Widgets
        self.record_button = ttk.Button(self.left_frame, text="Record (Shift+R)", command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, pady=5, sticky=tk.EW)

        self.pause_button = ttk.Button(self.left_frame, text="Pause", command=self.toggle_pause, state='disabled')
        self.pause_button.grid(row=1, column=0, pady=5, sticky=tk.EW)

        # Right Panel Widgets
        self.drop_area_label = ttk.Label(self.right_frame, text="Files to Transcribe")
        self.drop_area_label.grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.EW)

        self.drop_area = tk.Listbox(self.right_frame, selectmode=tk.MULTIPLE, height=10)
        self.drop_area.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.EW)

        self.clear_button = ttk.Button(self.right_frame, text="Clear Files", command=self.clear_files)
        self.clear_button.grid(row=2, column=0, pady=5, sticky=tk.EW)

        self.transcribe_button = ttk.Button(self.right_frame, text="Transcribe", command=self.transcribe_files)
        self.transcribe_button.grid(row=2, column=1, pady=5, sticky=tk.EW)

        self.progress = ttk.Progressbar(self.right_frame, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)

        self.output_area_label = ttk.Label(self.right_frame, text="Transcription Results")
        self.output_area_label.grid(row=4, column=0, columnspan=2, pady=(5, 0), sticky=tk.EW)

        self.output_area = tk.Listbox(self.right_frame, height=10)
        self.output_area.grid(row=5, column=0, columnspan=2, pady=5, sticky=tk.EW)

        self.right_frame.rowconfigure(1, weight=1)
        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.columnconfigure(1, weight=1)

        print("Widgets created.")

    def toggle_recording(self, event=None) -> None:
        """Toggles the recording state between start and stop."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> None:
        """Starts recording audio from the microphone."""
        self.is_recording = True
        self.is_paused = False
        self.record_button.config(text="Stop (Shift+R)")
        self.pause_button.config(state='normal', text="Pause")
        self.audio_data = []
        self.stop_recording_event = threading.Event()
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()
        print("Recording started.")

    def stop_recording(self) -> None:
        """Stops recording audio and initiates transcription."""
        self.is_recording = False
        self.is_paused = False
        self.record_button.config(text="Record (Shift+R)")
        self.pause_button.config(state='disabled', text="Pause")
        self.stop_recording_event.set()
        self.recording_thread.join()
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
        def callback(indata, frames, time_info, status):
            """Callback function to collect audio data."""
            if not self.is_paused:
                self.audio_data.append(indata.copy())

        with sd.InputStream(samplerate=self.samplerate, channels=1, callback=callback):
            while not self.stop_recording_event.is_set():
                time.sleep(0.1)

        print("Recording thread finished.")

    def save_and_transcribe_recording(self) -> None:
        """Saves the recorded audio to a file and transcribes it."""
        if not self.audio_data:
            messagebox.showwarning("Recording Error", "No audio data recorded.")
            return

        audio_data = np.concatenate(self.audio_data, axis=0)
        recorded_file_path = Path("recorded_audio.wav")
        sf.write(recorded_file_path, audio_data, self.samplerate)
        print(f"Audio saved to {recorded_file_path}")

        threading.Thread(target=self.transcribe_recorded_file, args=(recorded_file_path,)).start()

    def transcribe_recorded_file(self, file_path: Path) -> None:
        """Transcribes the recorded audio file and displays the transcript."""
        try:
            print(f"Transcribing recorded file: {file_path}")
            config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.nano)
            transcript = self.transcriber.transcribe(file_path.as_posix(), config=config)
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(transcript.error)

            self.display_transcript(transcript.text)
            print(f"Transcription completed for recorded file: {file_path}")
        except Exception as e:
            messagebox.showerror("Transcription Error", str(e))

    def display_transcript(self, text: str) -> None:
        """Displays the transcript in a new window with editing capabilities."""
        transcript_window = tk.Toplevel(self.root)
        transcript_window.title("Transcript")
        transcript_window.attributes('-topmost', True)  # Make window always on top

        # Store the original transcript
        self.previous_transcript = text

        # Transcript text widget
        self.text_widget = tk.Text(transcript_window, wrap='word')
        self.text_widget.insert(tk.END, text)
        self.text_widget.pack(expand=True, fill='both')

        # Additional Instructions Label
        instructions_label = ttk.Label(transcript_window, text="Additional Instructions:")
        instructions_label.pack()

        # Additional Instructions Text Widget
        self.instructions_widget = tk.Text(transcript_window, height=5, wrap='word')
        self.instructions_widget.pack(expand=False, fill='x')

        # Buttons Frame
        buttons_frame = ttk.Frame(transcript_window)
        buttons_frame.pack(pady=5)

        # Add "Tidy Up" button
        tidy_button = ttk.Button(
            buttons_frame,
            text="Tidy Up",
            command=self.tidy_transcript
        )
        tidy_button.grid(row=0, column=0, padx=5)

        # Add "Undo" button
        undo_button = ttk.Button(
            buttons_frame,
            text="Undo",
            command=self.undo_transcript
        )
        undo_button.grid(row=0, column=1, padx=5)

        # Modify "Copy" button to "Copy & Close"
        copy_close_button = ttk.Button(
            buttons_frame,
            text="Copy & Close (Ctrl+X)",
            command=lambda: self.copy_and_close_transcript(transcript_window)
        )
        copy_close_button.grid(row=0, column=2, padx=5)

        # Add keyboard shortcut for copy and close
        transcript_window.bind('<Control-x>', lambda event: self.copy_and_close_transcript(transcript_window))

    def tidy_transcript(self) -> None:
        """Tidies up the transcript using AI, with additional user instructions."""
        # Get the current text from the text widget
        transcript_text = self.text_widget.get("1.0", tk.END).strip()

        # Get additional instructions from the instructions widget
        additional_instructions = self.instructions_widget.get("1.0", tk.END).strip()

        # Define the assistant's base instructions
        assistant_instructions = (
            "You are an assistant that tidies up transcripts by fixing punctuation and capitalization, "
            "without changing the wording or meaning."
        )

        # Combine base instructions with additional instructions
        if additional_instructions:
            assistant_instructions += f" {additional_instructions}"

        try:
            # Set up the OpenAI client
            client = self.openai_client

            # Create an assistant
            assistant = client.beta.assistants.create(
                name="Transcript Tidy Assistant",
                instructions=assistant_instructions,
                model="gpt-4-1106-preview",
            )

            # Create a thread
            thread = client.beta.threads.create()

            # Send the transcript as a user message
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=transcript_text,
            )

            # Create and poll a run
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant.id,
                instructions="",
            )

            print("Run completed with status: " + run.status)

            if run.status == "completed":
                # Retrieve messages from the thread
                messages = client.beta.threads.messages.list(thread_id=thread.id)

                # Extract the assistant's reply
                for message in messages:
                    if message.role == "assistant":
                        # The content is a list of content parts
                        content_parts = message.content
                        # Extract the text content
                        reply = ""
                        for part in content_parts:
                            if part.type == "text":
                                reply += part.text.value

                        # Store the current transcript before updating
                        self.previous_transcript = transcript_text

                        # Update the text widget with the tidied-up text and mark changes
                        self.update_transcript_with_changes(transcript_text, reply)
                        break
            else:
                messagebox.showerror("Error", f"Run did not complete successfully. Status: {run.status}")

            # Delete the assistant to clean up
            client.beta.assistants.delete(assistant.id)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def update_transcript_with_changes(self, original: str, updated: str) -> None:
        """Updates the transcript text widget with the updated text and marks changes."""
        # Clear the text widget
        self.text_widget.delete("1.0", tk.END)

        # Define tags for formatting
        self.text_widget.tag_configure("addition", font=("Helvetica", 12, "bold"))
        self.text_widget.tag_configure("deletion", underline=True)

        # Use difflib to compare texts
        diff = difflib.ndiff(original.split(), updated.split())

        # Insert text with formatting
        for token in diff:
            code = token[:2]
            text = token[2:] + ' '  # Add space back

            if code == '  ':  # No change
                self.text_widget.insert(tk.END, text)
            elif code == '+ ':  # Addition
                self.text_widget.insert(tk.END, text, "addition")
            elif code == '- ':  # Deletion
                # Underline the previous character
                idx = self.text_widget.index(tk.END)
                if self.text_widget.index('1.0') != idx:
                    self.text_widget.tag_add("deletion", f"{idx} -1c", idx)
            else:
                self.text_widget.insert(tk.END, text)

    def undo_transcript(self) -> None:
        """Reverts the transcript to the previous version."""
        if self.previous_transcript:
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert(tk.END, self.previous_transcript)
        else:
            messagebox.showinfo("Undo", "No previous version to revert to.")

    def copy_and_close_transcript(self, window: tk.Toplevel, event=None) -> None:
        """Copies the transcript text to the clipboard and closes the window."""
        self.copy_transcript()
        window.destroy()

    def copy_transcript(self) -> None:
        """Copies the transcript text to the clipboard."""
        transcript_text = self.text_widget.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(transcript_text)

    def clear_files(self) -> None:
        """Clears the list of files to transcribe."""
        self.drop_area.delete(0, tk.END)
        print("Files cleared.")

    def drop_files(self, event: tk.Event) -> None:
        """Handles file drop events and adds valid files to the drop_area Listbox."""
        files = self.root.tk.splitlist(event.data)
        for file in files:
            if file.endswith(('.mp3', '.wav', '.mp4', '.m4a')):
                self.drop_area.insert(tk.END, file)
                print(f"File added: {file}")
            else:
                messagebox.showerror("Invalid File Type", f"Unsupported file type: {file}")

    def transcribe_files(self) -> None:
        """Transcribes the files listed in the drop_area Listbox."""
        self.transcribed_files = []
        files = self.drop_area.get(0, tk.END)
        if not files:
            messagebox.showwarning("No Files", "Please add files to transcribe.")
            return

        self.progress['value'] = 0
        self.progress['maximum'] = len(files)

        print("Transcription started.")

        # Start the transcription in a background thread
        threading.Thread(target=self._transcribe_files_background, args=(files,)).start()

    def _transcribe_files_background(self, files) -> None:
        """Runs the transcription process in a background thread."""
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.transcribe_file, file) for file in files]
            for future in futures:
                future.add_done_callback(self.update_progress)

    def transcribe_file(self, file_path: str) -> Path:
        """Transcribes a single file and writes the result to a text file."""
        file_path = Path(file_path)
        print(f"Transcribing file: {file_path}")
        transcript = self.transcriber.transcribe(file_path.as_posix())
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(transcript.error)

        output_path = file_path.with_suffix(".txt")
        with open(output_path, 'w') as f:
            f.write(transcript.text)

        # Use lock when accessing shared data
        with self.transcribed_files_lock:
            self.transcribed_files.append(output_path)

        print(f"Transcription completed for file: {file_path}")
        return output_path

    def update_progress(self, future: Future) -> None:
        """Updates the progress bar and adds a 'done' message to the output_area Listbox."""
        try:
            result = future.result()
            # Schedule the GUI updates to run in the main thread
            self.root.after(0, self._update_gui_after_transcription, result)
        except Exception as e:
            # Schedule the error message to run in the main thread
            self.root.after(0, messagebox.showerror, "Transcription Error", str(e))

    def _update_gui_after_transcription(self, result: Path) -> None:
        """Updates GUI elements after transcription is complete."""
        self.progress['value'] += 1
        self.output_area.insert(tk.END, f"Transcription done: {result}")
        print(f"Progress updated. Transcription done: {result}")
        if self.progress['value'] == self.progress['maximum']:
            self.create_xml_summary()


    def create_xml_summary(self) -> None:
        """Creates an XML file summarizing all transcribed files with their paths and contents."""
        if not self.transcribed_files:
            return

        root_element = ET.Element("Transcriptions")
        for file_path in self.transcribed_files:
            file_path = Path(file_path)
            with open(file_path, 'r') as f:
                transcript_text = f.read()

            transcription_element = ET.SubElement(root_element, "Transcription")
            ET.SubElement(transcription_element, "FilePath").text = str(file_path)
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
