import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.animation as animation
from pathlib import Path
from PIL import Image, ImageTk
import tensorflow as tf
import NST
from threading import Thread

Content_Image_path='default/Mona_Lisa.jpg'
Style_Image_path='default/Starry_Night.jpg'
Content_Image = None
Style_Image = None
Generated_Image = None
G_Image = None
Optimizer = 'Adam'
cost_history = []
process_on = False

def animate(i):
    ax.clear()
    ax.plot(range(len(cost_history)),cost_history)

def start_generation():
    global process_on
    try:
        Start_Button.config(text='Stop')
        status_bar.config(text=f'Starting...')
        m_n = Model.get()
        model = Model_list[m_n]()
        optimizer = Optimizer_list[Optimizer.get()](LearningRate.get())
        if m_n == "VGG19":
            STYLE_LAYERS = {
                'block1_conv1': 0.2,
                'block2_conv1': 0.2,
                'block3_conv1': 0.2,
                'block4_conv1': 0.2,
                'block5_conv1': 0.2,
            }
            CONTENT_LAYERS = 'block5_conv2' #'block4_conv2'
        else:
            STYLE_LAYERS = {
                'normal_concat_2': 0.5,
                'normal_concat_3': 0.5,
            }
            CONTENT_LAYERS = 'normal_concat_5'
        
        print(
            Content_Image_path,
            Style_Image_path,
            model,
            Model.get(),
            optimizer,
            Optimizer.get(),
            LearningRate.get(),
            CONTENT_LAYERS,
            STYLE_LAYERS,
            Alpha.get(),
            Beta.get(),
        )
        N = NST.NST(
            Content_Image_path=Content_Image_path,
            Style_Image_path=Style_Image_path,
            model=model,
            content_layer=CONTENT_LAYERS,
            style_layers=STYLE_LAYERS,
            alpha=Alpha.get(),
            beta=Beta.get(),
        )
        optimizer = tf.optimizers.Adam(2.0)
        it = Iterations.get()
        global cost_history
        cost_history = []
        for i in range(it):
            print("before")
            optimizer.minimize(N.total_cost,[N.Generated_Image])
            print("after")
            if i%1 == 0:
                losscost = N.total_cost()
                cost_history.append(losscost)
                print(i,losscost)
                Loss.config(text=f'Total Cost: {losscost}')
                status_bar.config(text=f'Processing... | Epoch: {i+1} | Total Cost: {losscost}')
                global Generated_Image
                global G_Image
                G_Image = N.as_image('G')
                Generated_Image = ImageTk.PhotoImage(G_Image)
                Generated_Image_panel.configure(image = Generated_Image)
            if not process_on:
                Start_Button.config(text='Stop')
                break

        Loss.config(text='Ready!')
        status_bar.config(text=f'Ready!')
        tf.compat.v1.reset_default_graph()
        process_on = False
        Start_Button.config(text='Start')

    except Exception as e:
        messagebox.showerror("Error",e)
        process_on = False
        Start_Button.config(text='Start')

def process_start_generation():
    global process_on
    if process_on:
        process_on = False
    else:
        process_on = True
        p = Thread(target=start_generation)
        p.start()

def open_image(image='C'):
    filepath = askopenfilename(
        filetypes = [
            ("Image files", ("*.png","*.jpg","*.jpeg")),
            ("All files", "*.*"),
        ]
    )
    if filepath:
        print(filepath, '->' , image)
        if image == 'C':
            global Content_Image
            Content_Image = ImageTk.PhotoImage(Image.open(filepath).resize((200,300)))
            Content_Image_panel.configure(image = Content_Image)
            global Content_Image_path
            Content_Image_path = filepath
            Content_Image_name.configure(text = Path(filepath).name)
            status_bar.config(text=f'Content Image Opened Successfully | Ready!')
        elif image == 'S':
            global Style_Image
            Style_Image = ImageTk.PhotoImage(Image.open(filepath).resize((200,300)))
            Style_Image_panel.configure(image = Style_Image)
            global Style_Image_path
            Style_Image_path = filepath
            Style_Image_name.configure(text = Path(filepath).name)
            status_bar.config(text=f'Style Image Opened Successfully | Ready!')
        else:
            messagebox.showerror("Value Error","Incorrect parameters")

def save_image():
    filepath = asksaveasfilename(
        filetypes = [
            ("Image files", ("*.png","*.jpg","*.jpeg")),
            ("All files", "*.*"),
        ],
        defaultextension='.png',
    )
    if filepath:
        try:
            status_bar.config(text=f'Saving Image...')
            print(filepath, '->' , 'Save')
            G_Image.save(filepath)
            status_bar.config(text=f'Image Saved | Ready!')
        except:
            messagebox.showerror("Error Saving Image",f"Unable to save {Path(filepath).name}")

### GUI #######################################################################
root = tk.Tk()
root.title('Neural Style Transfer')
root.resizable(False,False)
Variables_Frame = tk.Frame(root,border=2,relief=tk.GROOVE)
Button_Frame = tk.Frame(root,border=2,relief=tk.GROOVE)
Images_Frame = tk.Frame(root,border=2,relief=tk.GROOVE)
Chart_Frame = tk.Frame(root,border=2,relief=tk.GROOVE)
status_bar = tk.Label(root, text="Ready!", border=1, relief=tk.SUNKEN, anchor=tk.W)
Variables_Frame.grid(row=0,column=0,sticky='n')
Button_Frame.grid(row=1,column=0)
Images_Frame.grid(row=0,column=1,rowspan=2,sticky='nswe')
Chart_Frame.grid(row=0,column=2,rowspan=2,sticky='nswe')
status_bar.grid(row=2,column=0,columnspan=3,sticky='we')

#root.grid_rowconfigure([0,1,2], weight=1)
#root.grid_columnconfigure([0,1,2], weight=1)

### Main Menubar ##############################################################
menu = tk.Menu(root)
### File Menu #################################################################
filemenu = tk.Menu(menu)
filemenu.add_command(label="Open Content Image",command=lambda: open_image('C'))
filemenu.add_command(label="Open Style Image",command=lambda: open_image('S'))
filemenu.add_separator()
filemenu.add_command(label="Save Generated Image",command=save_image)
filemenu.add_separator()
filemenu.add_command(label="Exit",command=root.destroy)
### Assigning Menus ###########################################################
root.config(menu=menu)
menu.add_cascade(label="File",menu=filemenu)
### Model Dropdown ############################################################
Model_list = {
    "VGG19":tf.keras.applications.VGG19,
    "NASNetMobile":tf.keras.applications.NASNetMobile,
}
Model = tk.StringVar(Variables_Frame)
Model.set("VGG19")
Model_label = tk.Label(Variables_Frame,text="Model",anchor=tk.E)
Model_dropdown = tk.OptionMenu(Variables_Frame, Model, *list(Model_list.keys()))
Model_label.grid(row=0,column=0,sticky='we')
Model_dropdown.grid(row=0,column=1,sticky='we')
### Optimizer Dropdown ########################################################
Optimizer_list = {
    "Adam":tf.optimizers.Adam,
    "RMSprop":tf.optimizers.RMSprop,
}
Optimizer = tk.StringVar(Variables_Frame)
Optimizer.set("Adam")
Optimizer_label = tk.Label(Variables_Frame,text="Optimizer",anchor=tk.E)
Optimizer_dropdown = tk.OptionMenu(Variables_Frame, Optimizer, *list(Optimizer_list.keys()))
Optimizer_label.grid(row=1,column=0,sticky='we')
Optimizer_dropdown.grid(row=1,column=1,sticky='we')
### Learning Rate Entry #######################################################
LearningRate = tk.DoubleVar(Variables_Frame)
LearningRate.set(0.2)
LearningRate_label = tk.Label(Variables_Frame,text="Learning Rate",anchor=tk.E)
LearningRate_Entry = tk.Entry(Variables_Frame, textvariable = LearningRate)
LearningRate_label.grid(row=2,column=0,sticky='we')
LearningRate_Entry.grid(row=2,column=1,sticky='we')
### Iterations Entry ##########################################################
Iterations = tk.IntVar(Variables_Frame)
Iterations.set(100)
Iterations_label = tk.Label(Variables_Frame,text="Iterations", anchor=tk.E)
Iterations_Entry = tk.Entry(Variables_Frame, textvariable = Iterations)
Iterations_label.grid(row=3,column=0,sticky='we')
Iterations_Entry.grid(row=3,column=1,sticky='we')
### Alpha Entry ###############################################################
Alpha = tk.IntVar(Variables_Frame)
Alpha.set(5)
Alpha_label = tk.Label(Variables_Frame,text="Alpha", anchor=tk.E)
Alpha_Entry = tk.Entry(Variables_Frame, textvariable = Alpha)
Alpha_label.grid(row=4,column=0,sticky='we')
Alpha_Entry.grid(row=4,column=1,sticky='we')
### Beta Entry ################################################################
Beta = tk.IntVar(Variables_Frame)
Beta.set(1000)
Beta_label = tk.Label(Variables_Frame,text="Beta", anchor=tk.E)
Beta_Entry = tk.Entry(Variables_Frame, textvariable = Beta)
Beta_label.grid(row=5,column=0,sticky='we')
Beta_Entry.grid(row=5,column=1,sticky='we')
### Loss Status ###############################################################
Loss = tk.Label(Variables_Frame,text="Ready!")
Loss.grid(row=6,column=0,columnspan=2)
### Image Buttons #############################################################
C_Image_Button = tk.Button(Button_Frame, text="Open Content Image", command=lambda: open_image('C'))
S_Image_Button = tk.Button(Button_Frame, text="Open Style Image", command=lambda: open_image('S'))
G_Image_Button = tk.Button(Button_Frame, text="Save Generated Image", command=save_image)
Start_Button = tk.Button(Button_Frame, text="Start", command=process_start_generation)
C_Image_Button.grid(row=0,sticky='we')
S_Image_Button.grid(row=1,sticky='we')
G_Image_Button.grid(row=2,sticky='we')
Start_Button.grid(row=3,sticky='we')
### Content Image Display #####################################################
Content_Image = ImageTk.PhotoImage(Image.open(Content_Image_path).resize((200,300)))
Content_Image_panel = tk.Label(Images_Frame, image = Content_Image)
Content_Image_name = tk.Label(Images_Frame, text = Path(Content_Image_path).name)
Content_Image_panel.grid(row=0,column=0,sticky='nswe')
Content_Image_name.grid(row=1,column=0,sticky='nswe')
### Style Image Display #######################################################
Style_Image = ImageTk.PhotoImage(Image.open(Style_Image_path).resize((200,300)))
Style_Image_panel = tk.Label(Images_Frame, image = Style_Image)
Style_Image_name = tk.Label(Images_Frame, text = Path(Style_Image_path).name)
Style_Image_panel.grid(row=0,column=1,sticky='nswe')
Style_Image_name.grid(row=1,column=1,sticky='nswe')
### Generated Image Display ###################################################
Generated_Image = None
Generated_Image_panel = tk.Label(Images_Frame, image = Generated_Image)
Generated_Image_panel.grid(row=0,column=2,sticky='nswe')
###############################################################################
fig = Figure(figsize=(4,2), dpi=100)
ax = fig.add_subplot(111)
ax.plot([],[])
canvas = FigureCanvasTkAgg(fig, Chart_Frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
ani = animation.FuncAnimation(fig, animate, interval = 2000)
###############################################################################
root.mainloop()
