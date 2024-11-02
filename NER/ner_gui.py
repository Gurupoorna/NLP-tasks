import tkinter as tk
from tkinter import ttk
from NER import feats2vects, sent2features, svmclassifier
import re

def mock_predict(x):
    # Mock prediction; replace with `svmclassifier.predict(x)` in actual use.
    return [1 if token[0].isupper() else 0 for token in x]

# Function to perform NER on input sentence
def predict_ner():
    # Get user input
    user_sent = input_field.get("1.0", tk.END).strip()
    if not user_sent:
        return

    # Tokenize and prepare input
    # s_l = user_sent.split()
    pattern = r'(\s|"|:|,|:|;|\'|!|\?|\(|\)|\.$)'
    s_l = list(filter(lambda x : ('').__ne__(x) and (' ').__ne__(x), re.split(pattern , user_sent)))
    if s_l[-1] != '.':
        s_l.append('.')
    s_l = ['<START>'] + s_l + ['<STOP>']
    
    # Feature extraction and vectorization
    s2f = sent2features(s_l)
    x = feats2vects(s2f, test=True)
    
    # Prediction
    user_nei =  svmclassifier.predict(x)  # mock_predict(x)  # Replace mock_predict with svmclassifier.predict
    
    # Display tokens and NER tags
    output_field.delete(*output_field.get_children())  # Clear previous entries
    for w, e in zip(s_l, user_nei):
        tag = "NER" if e == 1 else ""
        output_field.insert('', 'end', values=(w, tag))

# Function to clear input and output fields
def clear_fields():
    input_field.delete("1.0", tk.END)
    output_field.delete(*output_field.get_children())

# Set up the main application window
root = tk.Tk()
root.title("NER GUI")
root.geometry("600x500")

# Input label and field
tk.Label(root, text="Enter Sentence:").pack(pady=5)
input_field = tk.Text(root, height=10, width=90)
input_field.pack()

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)
predict_button = tk.Button(button_frame, text="Predict NER", command=predict_ner)
predict_button.grid(row=0, column=0, padx=10)
clear_button = tk.Button(button_frame, text="Clear", command=clear_fields)
clear_button.grid(row=0, column=1, padx=10)

# Output label and treeview for displaying tokens and tags
tk.Label(root, text="NER Output:").pack(pady=5)
output_field = ttk.Treeview(root, columns=("Token", "NER Tag"), show="headings", height=10)
output_field.heading("Token", text="Token")
output_field.heading("NER Tag", text="NER Tag")
output_field.pack()

# Run the main loop
root.mainloop()
