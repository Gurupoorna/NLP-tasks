import tkinter as tk
from tkinter import messagebox
from main import initialize_hmm_tagger, pos_tag_sentence

class POSTaggingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("POS Tagging Interface")
        self.root.geometry("600x500")

        # Label asking for input
        self.label = tk.Label(root, text="Enter a sentence:", font=("Arial", 14))
        self.label.pack(pady=20)

        # Text input for the sentence
        self.text_input = tk.Entry(root, width=50, font=("Arial", 14))
        self.text_input.pack(pady=10)

        # Text box to display the POS-tagged output
        self.pos_output = tk.Text(root, height=10, width=70, font=("Arial", 12))
        self.pos_output.pack(pady=10)

        # Button to trigger POS tagging
        self.submit_button = tk.Button(root, text="Tag POS", command=self.display_pos_tags, font=("Arial", 14))
        self.submit_button.pack(pady=10)

        # Exit button to close the UI
        self.exit_button = tk.Button(root, text="Exit", command=self.root.quit, font=("Arial", 14))
        self.exit_button.pack(pady=10)

        # Initialize the HMM tagger using the function from main.py
        self.hmm_tagger, self.words = initialize_hmm_tagger()

    def display_pos_tags(self):
        # Get the sentence input from the text input box
        sentence = self.text_input.get()

        # Check if the input is empty
        if sentence.strip() == "":
            messagebox.showwarning("Input Error", "Please enter a valid sentence.")
            return

        # Use the pos_tag_sentence function from main.py to tag the sentence
        tagged_sentence = pos_tag_sentence(self.hmm_tagger, sentence, self.words)

        # Clear the output box before displaying new results
        self.pos_output.delete(1.0, tk.END)

        # Display the tagged sentence in the text box
        for word, tag in tagged_sentence:
            self.pos_output.insert(tk.END, f"{word}: {tag}\n")

# Run the Tkinter application
if __name__ == "__main__":
    root = tk.Tk()
    app = POSTaggingApp(root)
    root.mainloop()
