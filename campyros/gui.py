"""
To run the GUI, simply run:
import trajectory.gui
"""
import json
import ast
import numpy as np

import tkinter as tk
import tkinter.filedialog
from tkinter import ttk

from .main import LaunchSite, Parachute, Motor, Wind, Rocket
from .aero import RASAeroData

__copyright__ = """

    Copyright 2021 Daniel Gibbons

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

"""DATA AND CONSTANTS"""
# Dictionary of objects the user is allowed to add
OBJECTS_DICTIONARY = {
    "Launch Site": LaunchSite,
    "Parachute": Parachute,
    "Motor": Motor,
    "RASAeroData": RASAeroData,
    "Wind": Wind,
    "Rocket": Rocket,
}

# Possible inputs and the datatype, for each object. Values are in the format [datatype, default value]
LAUNCH_SITE_INPUTS = {
    "Rail length (m)": [float, 5],
    "Rail Yaw (deg)": [float, 0],
    "Rail Pitch (deg)": [float, 0],
    "Altitude (m)": [float, 0],
    "Longitude (deg)": [float, 0],
    "Latitude (deg)": [float, 0],
    "Variable Wind?": [bool],
    "Forecast plus time": [str, "016"],
    "Run Date (yyyymmdd)": [str, "20200101"],
    "Fast wind?": [bool],
    "Default wind [south, east, up] (m/s)": [list, [0, 0, 0]],
}
PARACHUTE_INPUTS = {"Not yet implemented": [str, ""]}
MOTOR_INPUTS = {"Not yet implemented": [str, ""]}
RASAERODATA_INPUTS = {"Not yet implemented": [str, ""]}
WIND_INPUTS = {"Not yet implemented": [str, ""]}
ROCKET_INPUTS = {"Not yet implemented": [str, ""]}

INPUTS_DICTIONARY = {
    "Launch Site": LAUNCH_SITE_INPUTS,
    "Parachute": PARACHUTE_INPUTS,
    "Motor": MOTOR_INPUTS,
    "RASAeroData": RASAERODATA_INPUTS,
    "Wind": WIND_INPUTS,
    "Rocket": ROCKET_INPUTS,
}

# Keep track of the objects and methods that the user has made:
current_objects = {}
current_methods = {}

# Keep track of the code
current_code = "import trajectory\n\n"

"""CLASSES"""
# Main window and functions
class Main:
    def __init__(self, master):
        self.master = master
        self.master.title("CUSF 6DOF Trajectory Simulator")

        # Menu
        menubar = tk.Menu(self.master)
        file_menu = tk.Menu(menubar, tearoff=False)

        menubar.add_cascade(label="File", menu=file_menu)
        self.master.config(menu=menubar)

        file_menu.add_command(label="Open", command=self.open)
        file_menu.add_command(label="Save", command=self.save)
        file_menu.add_command(label="Save as", command=self.save_as)
        file_menu.add_command(label="Generate code", command=self.generate_code)

        # Frames
        left_frame = tk.Frame(self.master, relief="sunken")
        left_frame.pack(side="left", fill="both")
        right_frame = tk.Frame(self.master, relief="sunken")
        right_frame.pack(side="right", expand="True", fill="both")

        # Tabs
        code_graphics_tab = ttk.Notebook(right_frame)
        graphics_tab = tk.Frame(code_graphics_tab, bg="white", relief="sunken")
        code_tab = tk.Frame(code_graphics_tab, bg="white", relief="sunken")
        code_graphics_tab.add(graphics_tab, text="Graphics")
        code_graphics_tab.add(code_tab, text="Code")
        code_graphics_tab.pack(expand=True, fill="both")

        objects_steps_tab = ttk.Notebook(left_frame)
        objects_tab = tk.Frame(objects_steps_tab)
        steps_tab = tk.Frame(objects_steps_tab)
        objects_steps_tab.add(objects_tab, text="Objects")
        objects_steps_tab.add(steps_tab, text="Steps")
        objects_steps_tab.pack(expand=True, fill="both")

        # Lists
        self.objects_list = ObjectsList(objects_tab)
        self.objects_list.pack(expand=True, fill="both")
        self.steps_list = StepsList(steps_tab)
        self.steps_list.pack(expand=True, fill="both")

        # Code tab text
        self.code_text = tk.Text(code_tab)
        self.code_text.insert("1.0", "The generated code is shown here")
        self.code_text.config(state="disabled")
        self.code_text.pack(expand=True, fill="both")

    def open(self):
        popup_message("'Open' is not yet implemented")

    def save(self):
        popup_message("'Save' is not yet implemented")

    def save_as(self):
        # Ask where you want to save the file
        save_file = tk.filedialog.asksaveasfile(
            defaultextension=".json", filetypes=[("JSON files", ".json")]
        )

        # Save current_objects and current_methods as a JSON
        with open(save_file.name, "w") as fp:
            json.dump(
                {
                    "current_objects": current_objects,
                    "current_methods": current_methods,
                },
                fp,
            )

        popup_message("Successfully saved file to {}".format(save_file.name))
        save_file.close()

    def generate_code(self):
        global current_code
        current_code = "import trajectory\n\n"

        object_names = list(current_objects.keys())
        for i in range(len(object_names)):
            object_inputs = current_objects[object_names[i]]
            code_to_add = object_to_code(
                object_names[i].replace(" ", "_"), object_inputs
            )
            current_code = current_code + code_to_add + "\n"

        refresh_code_display()
        popup_message("Finished generating code")


# List of objects
class ObjectsList(tk.Listbox):
    def __init__(self, parent):
        tk.Listbox.__init__(self, parent)
        self.popup_menu = tk.Menu(self.master, tearoff=0)
        self.popup_menu.add_command(label="Add object", command=self.add_object)
        self.popup_menu.add_command(label="Delete", command=self.delete_object)
        self.bind("<Button-3>", self.popup_menu_method)  # Button-2 on Aqua

    def popup_menu_method(self, event):
        try:
            self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            self.popup_menu.grab_release()

    def add_object(self):
        self.add_object_window = AddObjectWindow(self.master)
        self.add_object_window.attributes("-topmost", "true")

    def delete_object(self):
        for i in self.curselection()[::-1]:
            self.delete(i)


# List of methods/steps
class StepsList(tk.Listbox):
    def __init__(self, parent):
        tk.Listbox.__init__(self, parent)
        self.popup_menu = tk.Menu(self.master, tearoff=0)
        self.popup_menu.add_command(label="Add method", command=self.add_method)
        self.popup_menu.add_command(label="Delete", command=self.delete_method)
        self.bind("<Button-3>", self.popup_menu_method)  # Button-2 on Aqua

    def popup_menu_method(self, event):
        try:
            self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            self.popup_menu.grab_release()

    def add_method(self):
        self.add_method_window = AddMethodWindow(self.master)
        self.add_method_window.attributes("-topmost", "true")

    def delete_method(self):
        for i in self.curselection()[::-1]:
            self.delete(i)


# Window that opens to create a new object
class AddObjectWindow(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.title("Add object")

        # Set up the Listbox
        self.options_list = tk.Listbox(self, exportselection=False)
        for i in range(len(OBJECTS_DICTIONARY.keys())):
            self.options_list.insert(i + 1, list(OBJECTS_DICTIONARY.keys())[i])
        self.options_list.pack(side="left", expand=True, fill="both")

        # Inputs frame
        self.inputs_frame = tk.Frame(self)
        self.inputs_frame.pack(side="top", expand=True, fill="both")

        # When you click on an an item
        self.options_list.bind("<<ListboxSelect>>", self.show_inputs)

        # Frame at the bottom with the desired object name + an 'add object' button
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side="bottom")

        self.name_label = tk.Label(
            self.bottom_frame, text="Name"
        )  # Label saying 'Name'
        self.name_entry = tk.Entry(self.bottom_frame)  # Entry box for the name
        self.add_button = tk.Button(
            self.bottom_frame, text="Add object", command=self.add_object
        )  # Button to add the object

        self.name_label.grid(column=0, row=0)
        self.name_entry.grid(column=1, row=0)
        self.add_button.grid(column=2, row=0)

    def show_inputs(self, event):
        self.object_key = self.options_list.get(self.options_list.curselection())
        self.inputs = INPUTS_DICTIONARY[self.object_key]
        self.input_keys = list(self.inputs.keys())

        # Clear the frame of any old inputs
        for child in self.inputs_frame.winfo_children():
            child.destroy()

        # Add the input labels and entry boxes
        self.input_labels = []
        self.input_entries = []
        for i in range(len(self.input_keys)):
            self.input_labels.append(
                tk.Label(self.inputs_frame, text=self.input_keys[i])
            )
            self.input_labels[i].grid(column=0, row=i)

            datatype = self.inputs[self.input_keys[i]][0]
            if (
                datatype == float
                or datatype == int
                or datatype == str
                or datatype == list
            ):
                self.input_entries.append(tk.Entry(self.inputs_frame))
                self.input_entries[i].grid(column=1, row=i)

                default_value = self.inputs[self.input_keys[i]][1]
                self.input_entries[i].insert(0, str(default_value))

            elif datatype == bool:
                self.input_entries.append(TrueFalse(self.inputs_frame))
                self.input_entries[i].grid(column=1, row=i)
            else:
                self.input_entries.append(tk.Entry(self.inputs_frame))
                self.input_entries[i].grid(column=1, row=i)

    def add_object(self):
        # popup_message("The 'Add object' button is not yet functional")
        object_name = self.name_entry.get()
        if object_name == "":
            popup_message("You must input a name for the object")
            return None  # Break out of the function early

        dictionary_to_save = {}

        # Keep track of what kind of object we're saving
        dictionary_to_save["__OBJECT__"] = self.object_key

        # Save what the user entered for each input
        for i in range(len(self.input_entries)):
            datatype = self.inputs[self.input_keys[i]][0]

            if datatype == float or datatype == int or datatype == str:
                dictionary_to_save[self.input_keys[i]] = datatype(
                    self.input_entries[i].get()
                )

            elif datatype == bool:
                dictionary_to_save[self.input_keys[i]] = bool(
                    self.input_entries[i].var.get()
                )

            elif datatype == list:
                try:
                    literal_eval = ast.literal_eval(self.input_entries[i].get())
                except SyntaxError:
                    popup_message(
                        f"Failed to convert {self.input_entries[i].get()} to a list"
                    )
                    raise

                if type(literal_eval) is list:
                    dictionary_to_save[self.input_keys[i]] = literal_eval
                else:
                    popup_message(
                        f"Failed to convert {self.input_entries[i].get()} to a list"
                    )
                    raise TypeError(
                        f"Failed to convert {self.input_entries[i].get()} to a list"
                    )
            else:
                dictionary_to_save[self.input_keys[i]] = str(
                    self.input_entries[i].get()
                )

        # Save everything to the 'current_objects' dictionary
        current_objects[object_name] = dictionary_to_save
        refresh_objects_list()
        self.destroy()


# Window that opens when you want to add a new step/method
class AddMethodWindow(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.title("Add method")


# True of False radiobutton widget
class TrueFalse(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.var = tk.IntVar()
        self.RTrue = tk.Radiobutton(self, text="True", variable=self.var, value=1)
        self.RTrue.grid(column=0, row=0)
        self.RFalse = tk.Radiobutton(self, text="False", variable=self.var, value=0)
        self.RFalse.grid(column=1, row=0)


"""FUNCTIONS"""
# Call this function to make a popup message
def popup_message(message):
    error_window = tk.Toplevel(main_gui.master)
    error_window.title("Error")
    error_window.attributes("-topmost", "true")
    error_label = tk.Label(error_window, text=message)
    error_label.pack(side="top")
    ok_button = tk.Button(
        error_window, text="OK", command=lambda: error_window.destroy()
    )
    ok_button.pack(side="bottom")


# Call this function to refresh the objects list
def refresh_objects_list():
    objects_list = list(current_objects.keys())

    # Clear the list
    main_gui.objects_list.delete(0, "end")

    # Repopulate the list
    for i in range(len(objects_list)):
        main_gui.objects_list.insert(i + 1, objects_list[i])


def refresh_code_display():
    main_gui.code_text.config(state="normal")
    main_gui.code_text.delete("1.0", "end")
    main_gui.code_text.insert("1.0", current_code)
    main_gui.code_text.config(state="disabled")


# Takes the users inputs and returns the Python code needed to generate the object
def object_to_code(name, dictionary):
    if dictionary["__OBJECT__"] == "Launch Site":
        return (
            f"{name} = trajectory.LaunchSite("
            f'rail_length={dictionary["Rail length (m)"]}, '
            f'rail_yaw={dictionary["Rail Yaw (deg)"]}, '
            f'rail_pitch={dictionary["Rail Pitch (deg)"]}, '
            f'alt={dictionary["Altitude (m)"]}, '
            f'longi={dictionary["Longitude (deg)"]}, '
            f'lat={dictionary["Latitude (deg)"]}, '
            f'variable_wind={dictionary["Variable Wind?"]}, '
            f'forcast_plus_time="{dictionary["Forecast plus time"]}", '
            f'run_date="{dictionary["Run Date (yyyymmdd)"]}", '
            f'fast_wind={dictionary["Fast wind?"]} '
            f'default_wind={dictionary["Default wind [south, east, up] (m/s)"]})'
        )
    else:
        popup_message(
            "Converting the object type '{}' to code has not yet been implemented. This object has been skipped".format(
                dictionary["__OBJECT__"]
            )
        )


root = tk.Tk()
root.state("zoomed")
main_gui = Main(root)
root.mainloop()
