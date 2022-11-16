# -*- coding:utf-8 -*-
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from PIL import Image, ImageTk

sys.path.append("..")
import EmotionClassifierModel.Models

window = tk.Tk()
window.title("Emotion Recognition Application")
window.geometry("650x450+400+300")
window.resizable(0, 0)
window.attributes("-alpha", 0.9)
window.iconbitmap("imgs/Brain-icon.ico")

f = tk.StringVar()

img_dict = {-1: "imgs/-1.png", 0: "imgs/0.png", 1: "imgs/1.png"}
lable_dict = {-1: "negative", 0: "neutral", 1: "positive"}


def upload():
    f.set(
        filedialog.askopenfilename(
            title="OpenBCI output files",
            filetypes=[("csv", "*.csv"), ("All files", "*")],
        )
    )


def predict():
    if len(f.get()) == 0:
        messagebox.showinfo(title="Warning", message="You haven't upload file!")
        return
    EC = EmotionClassifierModel.Models.EmotionClassifier(True, f)
    EC.Init_train_test_data()
    match cbox.get():
        case "SVM":
            EC.SVM_model()
        case "AdaBoost":
            EC.AdaBoost_model()
        case "MLP":
            EC.MLP_model()
    result = EC.get_predicted_value()
    img = ImageTk.PhotoImage(Image.open(img_dict[result]))
    label_img = tk.Label(image=img)
    label_img.image = img
    label_img.place(anchor="center", x=325, y=135)
    tk.Label(window, text=lable_dict[result], font=("微软雅黑", 25)).place(
        anchor="center", x=325, y=280
    )


cbox = ttk.Combobox(window)
cbox["value"] = ("SVM", "AdaBoost", "MLP")
cbox.current(0)
cbox.grid(row=1, sticky="NW")

tk.Label(window, textvariable=f, font=("微软雅黑", 13)).place(anchor="center", x=325, y=330)
tk.Button(window, text="Upload and predict", width=15, height=3, command=upload).place(
    anchor="center", x=163, y=385
)
tk.Button(window, text="Upload and predict", width=15, height=3, command=predict).place(
    anchor="center", x=487, y=385
)


window.mainloop()
