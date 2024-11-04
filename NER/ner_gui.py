import tkinter as tk
from tkinter import ttk
from tkinter import font
from NER import feats2vects, sent2features, svmclassifier, my_token_preps

def mock_predict(x):
    # Mock prediction; replace with `svmclassifier.predict(x)` in actual use.
    return [1 if (token.istitle() and (pos in ['NN', 'NNP'])) or (pos in ['NNP']) else 0 for token, pos in x]

# Function to perform NER on input sentence
def predict_ner():
    # Get user input
    user_sent = input_field.get("1.0", tk.END).strip()
    if not user_sent:
        return

    # Tokenize and prepare input
    s_l = my_token_preps(user_sent)
    
    # Feature extraction and vectorization
    s2f = sent2features(s_l)
    x = feats2vects(s2f, test=True)
    
    # Prediction
    user_nei =  svmclassifier.predict(x)  # mock_predict(x)  # Replace mock_predict with svmclassifier.predict
    ume = mock_predict(s_l)
    
    # Display tokens and NER tags
    output_field.delete(*output_field.get_children())  # Clear previous entries
    insert_colored_row([(w,p,e,k) for (w,p), e, k in zip(s_l, user_nei, ume)])

# Function to insert color-coded rows
def insert_colored_row(data):
    for i, (token, pos_tag, ner_tag, k) in enumerate(data):
        tag = "oddrow" if i % 2 == 0 else "evenrow"
        ner_tag = "NER" if ner_tag == 1 else ""
        output_field.insert("", "end", values=(token, pos_tag, ner_tag, k), tags=(tag,))

# Function to clear input and output fields
def clear_fields():
    input_field.delete("1.0", tk.END)
    output_field.delete(*output_field.get_children())

# Set up the main application window
root = tk.Tk()
root.title("NER GUI")
root.geometry("700x550")

# Define font style and size
font_style = font.Font(family="Helvetica", size=14)
button_font = font.Font(family="Helvetica", size=12, weight="bold")
treeview_font = font.Font(family="Helvetica", size=12)

# Input label and field with updated font
tk.Label(root, text="Enter Sentence:", font=font_style).pack(pady=5)
input_field = tk.Text(root, height=5, width=60, font=font_style)
input_field.pack(pady=(0, 15))

# Buttons with updated font
button_frame = tk.Frame(root)
button_frame.pack(pady=10)
predict_button = tk.Button(button_frame, text="Predict NER", command=predict_ner, font=button_font)
predict_button.grid(row=0, column=0, padx=10)
clear_button = tk.Button(button_frame, text="Clear", command=clear_fields, font=button_font)
clear_button.grid(row=0, column=1, padx=10)

# Output label
tk.Label(root, text="NER Output:", font=font_style).pack(pady=5)

# Frame for output field and scrollbar
output_frame = tk.Frame(root)
output_frame.pack(fill="both", expand=True)

# Scrollbar
scrollbar = tk.Scrollbar(output_frame)
scrollbar.pack(side="right", fill="y")

# Treeview for displaying tokens and tags with larger font
output_field = ttk.Treeview(output_frame, columns=("Token", "POS Tag", "NER Tag", 'ku'), show="headings", height=10, yscrollcommand=scrollbar.set)
output_field.heading("Token", text="Token", anchor="w")
output_field.heading("POS Tag", text="POS Tag", anchor="w")
output_field.heading("NER Tag", text="NER Tag", anchor="w")
output_field.heading("ku", text="", anchor="w")
output_field.column("Token", anchor="w", width=180)
output_field.column("POS Tag", anchor="w", width=120)
output_field.column("NER Tag", anchor="w", width=120)
output_field.column("ku", anchor="w", width=12)
output_field.pack(fill="both", expand=True)

# Configure the scrollbar
scrollbar.config(command=output_field.yview)

# Apply the font to the Treeview rows
style = ttk.Style()
style.configure("Treeview", font=treeview_font, rowheight=30)
style.configure("Treeview.Heading", font=button_font)

# Alternate row colors
style.map("Treeview", background=[("selected", "#DDEEFF")])
output_field.tag_configure("oddrow", background="#f0f0f0")
output_field.tag_configure("evenrow", background="#ffffff")

# Run the main loop
root.mainloop()