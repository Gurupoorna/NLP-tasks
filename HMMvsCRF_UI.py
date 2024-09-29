import nltk

from nltk.tokenize import word_tokenize
from CRF.CRFTagger import initialize_crf_tagger
from HMM.HMM import initialize_hmm_tagger
import tkinter as tk
from tkinter import scrolledtext
from tkinter import font as tkfont
from numpy import load

TAG_COLORS = {
    'NOUN': 'blue',
    'VERB': 'green',
    'ADJ': 'orange',
    'ADV': 'purple',
    'PRON': 'darkred',
    'DET': 'brown',
    'ADP': 'cyan',
    'CONJ': 'magenta',
    'NUM': 'red',
    'PRT': 'teal',
    '.': 'black',
    'X': 'gray',
}

def tag_sentence():
    sentence = input_text.get("1.0", tk.END).strip()
    if sentence:
        predicted_hmm_tags = hmm_tagger.tag(sentence)
        predicted_crf_tags = crf_tagger.tag(sentence)
        correct_tags = nltk.pos_tag(word_tokenize(sentence), tagset='universal')
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Word"+" "*9+"-> HMM's pred | CRF's pred | nltk POS Tag\n", 'heading')

        mismatches = []
        for ((word, hmm_tag), crf_tag, (_, corr_tag)) in zip(predicted_hmm_tags, predicted_crf_tags, correct_tags):
            # pred_tag_color = TAG_COLORS.get(hmm_tag, 'black')
            # corr_tag_color = TAG_COLORS.get(corr_tag, 'black')
            result_text.insert(tk.END, f"{word:16s}->", 'word')
            result_text.insert(tk.END, f"    {hmm_tag:5s}   ", f"pred_{hmm_tag}")
            result_text.insert(tk.END, f"|   {crf_tag:5s}   ", f"pred_{crf_tag}")
            result_text.insert(tk.END, f"|   {corr_tag:5s}\n", f"corr_{corr_tag}")
            if hmm_tag != crf_tag or crf_tag != corr_tag or corr_tag != hmm_tag:
                mismatches.append((word, hmm_tag, crf_tag, corr_tag))

        if mismatches:
            result_text.insert(tk.END, "\nMismatches Found b/w HMM & CRF & nltk predictions:\n", 'heading')
            for word, hmm_tag, crf_tag, corr_tag in mismatches:
                result_text.insert(tk.END, f"{word:18s}:  {hmm_tag:5s}  -  {crf_tag:5s}  -  {corr_tag:5s}\n", 'mismatch')
        else:
            result_text.insert(tk.END, "\nNo Mismatches Found.\n", 'heading')
vals = load('HMM/hmm_probs.npz')
hmm_tagger, words, pos_tags = initialize_hmm_tagger(A=vals['A'], B=vals['B'], Pi=vals['Pi'])
crf_tagger = initialize_crf_tagger('crf_UI', path='CRF/crf_UI.crfsuite')

root = tk.Tk()
root.title("HMM v/s CRF POS Taggers")

heading_font = tkfont.Font(family="Consolas", size=22, weight="bold")
body_font = tkfont.Font(family="Courier", size=20)
button_font = tkfont.Font(family="Helvetica", size=18, weight="bold")

input_label = tk.Label(root, text="Enter Sentence:", font=heading_font)
input_label.pack(pady=(10, 0))

input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=5, font=body_font)
input_text.pack(pady=(5, 10))

tag_button = tk.Button(root, text="Tag Sentence", command=tag_sentence, bg='#4CAF50', fg='white', font=button_font, padx=20, pady=5)
tag_button.pack(pady=(0, 10))

result_label = tk.Label(root, text="Results:", font=heading_font)
result_label.pack()

result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15, font=body_font)
result_text.pack(pady=(5, 10))

result_text.tag_configure('heading', font=heading_font)
result_text.tag_configure('word', font=body_font)
result_text.tag_configure('mismatch', font=body_font, foreground='red')

for tag, color in TAG_COLORS.items():
    result_text.tag_configure(f"pred_{tag}", font=body_font, foreground=color)
    result_text.tag_configure(f"corr_{tag}", font=body_font, foreground=color)

root.mainloop()