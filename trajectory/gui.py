import tkinter as tk
from tkinter import ttk

#Dictionary of objects the user is allowed to add
OBJECTS_DICTIONARY = {"Launch Site" : None, "Rocket" : None}

#Main window and functions
class Main:
    def __init__(self, master):
        self.master = master
        self.master.title("CUSF 6DOF Trajectory Simulator")
        self.init_UI()
        
    def init_UI(self):
        #Menu
        menubar = tk.Menu(self.master)
        file_menu = tk.Menu(menubar, tearoff=False)
        
        menubar.add_cascade(label="File", menu=file_menu)
        self.master.config(menu=menubar)
        
        file_menu.add_command(label="Save as", command=self.save_as)
        file_menu.add_command(label="Generate code", command=self.generate_code)
        
        #Frames
        left_frame = tk.Frame(self.master, relief="sunken")
        left_frame.pack(side="left", fill='both')
        right_frame = tk.Frame(self.master, relief="sunken")
        right_frame.pack(side="right", expand='True', fill="both")
        
        #Tabs
        code_graphics_tab = ttk.Notebook(right_frame)
        graphics_tab = tk.Frame(code_graphics_tab, bg='white', relief='sunken')
        code_tab = tk.Frame(code_graphics_tab, bg='white', relief='sunken')
        code_graphics_tab.add(graphics_tab, text="Graphics")
        code_graphics_tab.add(code_tab, text="Code")
        code_graphics_tab.pack(expand = True, fill='both')

        objects_steps_tab = ttk.Notebook(left_frame)
        objects_tab = tk.Frame(objects_steps_tab)
        steps_tab = tk.Frame(objects_steps_tab)
        objects_steps_tab.add(objects_tab, text="Objects")
        objects_steps_tab.add(steps_tab, text="Steps")
        objects_steps_tab.pack(expand = True, fill='both')  
        
        #Lists
        self.objects_list = ObjectsList(objects_tab)
        self.objects_list.pack(expand=True, fill='both')
        self.steps_list = StepsList(steps_tab)
        self.steps_list.pack(expand=True, fill='both')
        
        #Code tab text
        code_text = tk.Text(code_tab)
        code_text.insert('1.0', "The generated code is shown here")
        code_text.config(state='disabled')
        code_text.pack(expand=True, fill='both')
        
    def save_as(self):
        print("'Save as' is not yet implemented")
        
    def generate_code(self):
        print("'Generate code' is not yet functional")

#List of objects
class ObjectsList(tk.Listbox):
    def __init__(self, parent):
        tk.Listbox.__init__(self, parent)
        self.popup_menu = tk.Menu(self.master, tearoff=0)
        self.popup_menu.add_command(label="Add object", command=self.add_object)
        self.popup_menu.add_command(label="Delete", command=self.delete_object)
        self.bind("<Button-3>", self.popup_menu_method) # Button-2 on Aqua

    def popup_menu_method(self, event):
        try:
            self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            self.popup_menu.grab_release()
    
    def add_object(self):
        self.add_object_window = AddObjectWindow(self.master)
        self.add_object_window.attributes('-topmost', 'true')

    def delete_object(self):
        for i in self.curselection()[::-1]:
            self.delete(i)

#List of methods/steps
class StepsList(tk.Listbox):
    def __init__(self, parent):
        tk.Listbox.__init__(self, parent)
        self.popup_menu = tk.Menu(self.master, tearoff=0)
        self.popup_menu.add_command(label="Add method", command=self.add_method)
        self.popup_menu.add_command(label="Delete", command=self.delete_method)
        self.bind("<Button-3>", self.popup_menu_method) # Button-2 on Aqua

    def popup_menu_method(self, event):
        try:
            self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            self.popup_menu.grab_release()
    
    def add_method(self):
        self.add_method_window = AddMethodWindow(self.master)
        self.add_method_window.attributes('-topmost', 'true')

    def delete_method(self):
        for i in self.curselection()[::-1]:
            self.delete(i)

#Window that opens to create a new object   
class AddObjectWindow(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.title("Add object")

        self.options_list = tk.Listbox(self)
        for i in range(len(OBJECTS_DICTIONARY.keys())):
            self.options_list.insert(i+1, list(OBJECTS_DICTIONARY.keys())[i])
        self.options_list.pack(side="left", expand=True, fill="y")
        self.description = tk.Text(self)
        self.description.pack(side="right", expand=True, fill="both")
        self.description.config(state='disabled')

    def show_description(self):
        pass


#Window that opens when you want to add a new step/method
class AddMethodWindow(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.title("Add method")
        
root = tk.Tk()
root.state('zoomed')
my_gui = Main(root)
root.mainloop()