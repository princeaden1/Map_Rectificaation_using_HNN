import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk,ImageDraw
import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class MapClass(tk.Tk):
    def __init__(self, map_arg):
        super().__init__()
        self.title("Map Editor")
        self.geometry("969x857")
        
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.im = Image.open(map_arg)
        self.image_tk = ImageTk.PhotoImage(self.im)
        self.zoom = 1
        
        self.shape_array = ['Dot', 'Line', 'Rectangle']
        self.shape_string = tk.StringVar()
        self.shape_string.set('Dot')



        #canvas main frame
        frame_main = tk.Frame(self)
        frame_main.grid(row=0, column=0, sticky="nsew")
        #create canvas for displaying

        
        self.canvas = tk.Canvas(frame_main, width=950, height=740)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # self.canvas.pack()
        
        
            # add zoom buttons
        self.zoom_in_btn = tk.Button(frame_main, text="+", command=self.zoom_in)
        self.zoom_in_btn.place(x=900, y=50, anchor=tk.NW)

        self.zoom_out_btn = tk.Button(frame_main, text="-", command=self.zoom_out)
        self.zoom_out_btn.place(x=900, y=90, anchor=tk.NW)


        #add new frame for scrolling
        frame_secondary = ttk.Frame(self)
        frame_secondary.grid(row=1, column=0, padx=10, pady=10)
        # Create horizontal scrollbar
        self.horiz_scroll = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.horiz_scroll.set)
        self.horiz_scroll.grid(row=1, column=0, sticky="nsew")
        # Create vertical scrollbar
        self.vert_scroll = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vert_scroll.set)
        self.vert_scroll.grid(row=0, column=1, sticky="nsew")
        # Calculate center coordinates
        x = self.canvas.winfo_reqwidth() / 2
        y = self.canvas.winfo_reqheight() / 2
        # Display the image centered on the canvas
        self.canvas.create_image(x, y, anchor=tk.CENTER, image=self.image_tk)
        # Add zoom functionality
        self.canvas.bind("<MouseWheel>", self.zoom_image)




        




        # Create a canvas
        # self.canvas = tk.Canvas(self, scrollregion=(0, 0, 2000, 2000))
        # self.canvas.grid(row=0, column=0, sticky="nsew")
        # self.canvas.pack(fill=tk.BOTH, expand=True)

        frame = ttk.Frame(self)
        frame.grid(row=2, column=0, sticky="nsew", pady=10)
        # Info pane
        info_frame = ttk.Frame(frame, relief=tk.SOLID, borderwidth=2)
        info_frame.grid(row=0, column=0, sticky="nsew")
        

        self.mini_map_canvas = tk.Canvas(info_frame, width=100, height=100)
        self.mini_map_canvas.pack(side="left")
        self.update_mini_map()


        






        # Tool widgets
        draw_frame = ttk.Frame(frame)
        draw_frame.grid(row=0, column=1, padx=(10,0))

        ttk.Label(draw_frame, text="Drawing Tool", justify='left').grid(row=0, column=0, sticky="w")
        self.color_combo = ttk.Combobox(draw_frame, textvariable=self.shape_string,
            values=self.shape_array)
        self.color_combo.grid(row=1, column=0) 
        self.color_combo.bind("<<ComboboxSelected>>")
        frame3 = ttk.Frame(draw_frame)
        frame3.grid(row=2, column=0, sticky="nsew")
        self.draw_shape = ttk.Button(frame3, text="Draw", command=self.handle_drawing)
        self.draw_shape.grid(row=0, column=0)
        self.erase_shape = ttk.Button(frame3, text="Erase", command=self.handle_erase)
        self.erase_shape.grid(row=0, column=1, padx=(18,0))



        # zoom widgets
        # ttk.Label(frame, text="Zoom").grid(row=0, column=4)
        # self.zoom_combo = ttk.Combobox(frame, state="readonly", values=["100%","200%","300%"])
        # self.zoom_combo.grid(row=0, column=5)

        # Buttons

        button_frame_1 = ttk.Frame(frame)
        button_frame_1.grid(row=0, column=6, padx=(10,0))

        self.apply_hopfield = ttk.Button(button_frame_1, text="Load File", command=self.open_file)
        self.apply_hopfield.grid(row=0, column=6)

        self.load_file = ttk.Button(button_frame_1, text="Rectify", command=self.open_rectify)
        self.load_file.grid(row=1, column=6)
        
        


        button_frame_2 = ttk.Frame(frame)
        button_frame_2.grid(row=0, column=7, padx=(20,0))


        self.save_button = ttk.Button(button_frame_2, text="Save", command=self.save_image)
        self.save_button.grid(row=0, column=0)

        
        self.close_button = ttk.Button(button_frame_2, text="Exit", command=self.close_window)
        self.close_button.grid(row=1, column=0)

        # Status bar 
        self.statusbar = ttk.Label(self)
        self.statusbar.grid(row=3, column=0, sticky="nsew")

        # Initialize drawing variables
        self.drawing = False
        self.start_x, self.start_y = None, None
        self.tool = None
        self.trace_line = None
        self.trace_rect = None
        self.image_x_co = None
        self.image_y_co = None
        # self.show_image()

        
    def save_image(self):
        # Create the output folder if it doesn't exist
        output_folder = "output"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the image inside the output folder
        file_path = os.path.join(output_folder, "map.pgm")
        self.im.save(file_path)


    def zoom_in(self):
        self.zoom *= 1.2
        self.show_image()
        
    def zoom_out(self):
        self.zoom /= 1.2
        self.show_image()

        
    def zoom_image(self, event):
        # Zoom in or out based on mouse wheel direction
        if event.delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1
        self.show_image()


    def show_image(self):

        
        
        if self.im.mode != 'L':
            self.im = self.im.convert('L')

        self.map_width_cells = self.im.size[0]
        self.map_height_cells = self.im.size[1]
        # self.im = self.im.point(lambda x: x * 1.5)

        self.min_multiplier = math.ceil(self.canvas.winfo_width() / self.map_width_cells)
        self.pixels_per_cell = int(self.min_multiplier * self.zoom)

        # Resize the image with antialiasing
        new_width = int(self.map_width_cells * self.pixels_per_cell)
        new_height = int(self.map_height_cells * self.pixels_per_cell)

        if new_width>0 and new_height>0:
            resized_image = self.im.resize((new_width, new_height), Image.LANCZOS)
        else:
            resized_image = self.im
        # resized_image = self.im.resize((, self.map_height_cells * self.pixels_per_cell))

        self.image_tk = ImageTk.PhotoImage(resized_image)


        self.image_x_co = (self.canvas.winfo_width() / 2) - (resized_image.width / 2)
        self.image_y_co = (self.canvas.winfo_height() / 2) - (resized_image.height / 2)

        # Update the canvas
        self.canvas.create_image(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                                anchor=tk.CENTER, image=self.image_tk)
        
        self.canvas.configure(scrollregion=(0, 0, int(self.map_width_cells * self.pixels_per_cell), int(self.map_height_cells * self.pixels_per_cell)))


        # if(self.zoom>1.5):
        if(self.zoom>1.5):
                # Limit the grid to the area covered by the image
            grid_width = int(self.map_width_cells * self.pixels_per_cell)
            grid_height = int(self.map_height_cells * self.pixels_per_cell)

            for x in range(0, grid_width, self.pixels_per_cell):
                self.canvas.create_line(x, 0, x, grid_height, fill='lightgray')

            for y in range(0, grid_height, self.pixels_per_cell):
                self.canvas.create_line(0, y, grid_width, y, fill='lightgray')

        self.update_mini_map()


    

    
    def handle_erase(self):
        self.pencil = "white"
        self.handle_draw()
    def handle_drawing(self):
        self.pencil = "black"
        self.handle_draw()


    def close_window(self):
        # Close the window
        self.destroy()

    def handle_draw(self):
        self.tool = self.shape_string.get()
        # Unbind all drawing events
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        if self.tool == 'Dot':
            self.canvas.bind("<Button-1>", self.draw_dot)
        elif self.tool == 'Line':
            self.canvas.bind("<Button-1>", self.start_line)
            self.canvas.bind("<B1-Motion>", self.draw_line)
            self.canvas.bind("<ButtonRelease-1>", self.end_line)
        elif self.tool == 'Rectangle':
            self.canvas.bind("<Button-1>", self.start_rect)
            self.canvas.bind("<B1-Motion>", self.draw_rect)
            self.canvas.bind("<ButtonRelease-1>", self.end_rect)

    def draw_dot(self, event):
        x, y = event.x, event.y
        x_coord, y_coord = self.canvas_to_image_coords(x, y)
        self.draw = ImageDraw.Draw(self.im)
        self.draw.ellipse((x_coord - 0.1, y_coord - 0.1, x_coord + 0.1, y_coord + 0.1), fill=self.pencil)
        self.show_image()

    def start_line(self, event):
        self.start_x, self.start_y = event.x, event.y

    def draw_line(self, event):
        if self.trace_line:
            self.canvas.delete(self.trace_line)
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.trace_line = self.canvas.create_line(self.canvas.canvasx(self.start_x), self.canvas.canvasy(self.start_y), x, y, fill=self.pencil)


    def end_line(self, event):
        self.canvas.delete(self.trace_line)
        x, y = event.x, event.y
        start_x_coord, start_y_coord = self.canvas_to_image_coords(self.start_x, self.start_y)
        end_x_coord, end_y_coord = self.canvas_to_image_coords(x, y)
        self.draw = ImageDraw.Draw(self.im)
        self.draw.line((start_x_coord, start_y_coord, end_x_coord, end_y_coord), fill=self.pencil, width=1)
        self.show_image()

    def start_rect(self, event):
        self.start_x, self.start_y = event.x, event.y

    def draw_rect(self, event):
        if self.trace_rect:
            self.canvas.delete(self.trace_rect)
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.trace_rect = self.canvas.create_rectangle(self.canvas.canvasx(self.start_x), self.canvas.canvasy(self.start_y), x, y, outline=self.pencil)


    def end_rect(self, event):
        self.canvas.delete(self.trace_rect)
        x, y = event.x, event.y
        start_x_coord, start_y_coord = self.canvas_to_image_coords(self.start_x, self.start_y)
        end_x_coord, end_y_coord = self.canvas_to_image_coords(x, y)
        self.draw = ImageDraw.Draw(self.im)
        self.draw.rectangle((start_x_coord, start_y_coord, end_x_coord, end_y_coord), outline=self.pencil)
        self.show_image()

    def canvas_to_image_coords(self, x, y):
        canvas_x = self.canvas.canvasx(x)
        canvas_y = self.canvas.canvasy(y)
        
        if self.image_x_co!=None:
            image_x = int((canvas_x - self.image_x_co) / self.pixels_per_cell)
            image_y = int((canvas_y - self.image_y_co) / self.pixels_per_cell)
            return image_x, image_y
        else:
            # self.zoom_in()
            return canvas_x, canvas_y
        
    
    # open dialog box to select file
    def open_rectify(self):
        # Create the output folder if it doesn't exist
        output_folder = "ogm_rectier"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Save the image inside the output folder
        file_path = os.path.join(output_folder, "map.pgm")
        self.im.save(file_path)
        main(self)
            
            
    # open dialog box to select file
    def open_file(self):
        # Open a file dialog to select an image file
        path = filedialog.askopenfilename(filetypes=[("Image Files", ".pgm")])
        if path:
            # Load the image
            self.im = Image.open(path)
            self.show_image()
            





    def update_mini_map(self):
        # Clear previous contents
        self.mini_map_canvas.delete("all")

        # Create a thumbnail of the original image to fit within the mini-map canvas
        thumbnail = self.im.copy()
        thumbnail.thumbnail((100, 100))

        # Display the thumbnail as the background of the mini-map canvas
        self.mini_map_image = ImageTk.PhotoImage(thumbnail)
        self.mini_map_canvas.create_image(0, 0, anchor="nw", image=self.mini_map_image)

        # Calculate the position and size of the viewport box
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        viewport_width = canvas_width / self.zoom
        viewport_height = canvas_height / self.zoom
        viewport_x = self.canvas.canvasx(0) / self.zoom
        viewport_y = self.canvas.canvasy(0) / self.zoom
        viewport_x2 = viewport_x + viewport_width
        viewport_y2 = viewport_y + viewport_height

        # Adjust viewport coordinates according to the zoom level
        viewport_x /= self.im.width
        viewport_y /= self.im.height
        viewport_x2 /= self.im.width
        viewport_y2 /= self.im.height

        # Draw viewport box on mini map
        box_x = int(viewport_x * 100)
        box_y = int(viewport_y * 100)
        box_x2 = int(viewport_x2 * 100)
        box_y2 = int(viewport_y2 * 100)
        self.red_rectangle = self.mini_map_canvas.create_rectangle(box_x, box_y, box_x2, box_y2, outline="red")

        # Make the entire mini map canvas draggable
        self.mini_map_canvas.bind("<Button-1>", self.on_drag_start)
        self.mini_map_canvas.bind("<B1-Motion>", self.on_drag)
        
        # Simulate a click event on the mini map canvas to make it active on page load
        self.mini_map_canvas.event_generate("<Button-1>", x=box_x, y=box_y)




    def on_drag_start(self, event):
        # Record the starting position of the drag
        self.start_x = event.x
        self.start_y = event.y

    def on_drag(self, event):
        # Calculate the change in position
        delta_x = event.x - self.start_x
        delta_y = event.y - self.start_y

        # Update the position of the red rectangle
        self.mini_map_canvas.move(self.red_rectangle, delta_x, delta_y)

        # Update the view of self.canvas
        x = self.mini_map_canvas.coords(self.red_rectangle)[0]
        y = self.mini_map_canvas.coords(self.red_rectangle)[1]
        self.canvas.xview_moveto(x / 100)
        self.canvas.yview_moveto(y / 100)

        # Update starting position for next drag
        self.start_x = event.x
        self.start_y = event.y








def load_and_preprocess_templates(template_paths, target_size):
    templates = []
    for path in template_paths:
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        template = cv2.resize(template, target_size, interpolation=cv2.INTER_AREA)
        template = (template > 128).astype(np.uint8)  # Binarize the image
        templates.append(template)
    return templates

def calculate_similarity(pattern1, pattern2):
    return np.sum(pattern1 == pattern2)

def main(self):
    # Load and preprocess template images
    template_paths = ["temp/h_w.jpg", "temp/ver_white.jpg", "temp/hor.jpg", "temp/ver.jpg", "temp/bc.png"]
    target_size = (3, 3)  # Desired size for all templates
    templates = load_and_preprocess_templates(template_paths, target_size)

    # Create Hopfield network and train with templates
    pattern_size = target_size
    hopfield_net = HopfieldNetwork(pattern_size)
    hopfield_net.train(templates)

    # Load the input image
    input_image = cv2.imread("ogm_rectier/map.pgm", cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if input_image is None:
        print("Error: Unable to load the input image.")
        return

    # Thresholding: Binarize the input image using Otsu's method
    _, binary_map = cv2.threshold(input_image, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Hopfield Network Processing: Divide the binary map into blocks and process each block with HNN
    block_size = pattern_size
    num_blocks_x = binary_map.shape[0] // block_size[0]
    num_blocks_y = binary_map.shape[1] // block_size[1]

    rectified_map = np.ones_like(binary_map, dtype=np.uint8) * 255  # Initialize with white background

    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            block = binary_map[i * block_size[0]:(i + 1) * block_size[0],
                               j * block_size[1]:(j + 1) * block_size[1]]

            best_similarity = -1
            best_pattern = None
            for template in templates:
                similarity = calculate_similarity(block, template)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pattern = template

            converged_block = hopfield_net.update(best_pattern)
            # Set unoccupied cells to white and occupied cells to black
            rectified_map[i * block_size[0]:(i + 1) * block_size[0],
                          j * block_size[1]:(j + 1) * block_size[1]] = (1 - converged_block) * 255

    # Border Reconstruction: Reconstruct borders and walls in the rectified map
    wall_thickness = 2  # User-defined wall thickness parameter
    for i in range(rectified_map.shape[0]):
        for j in range(rectified_map.shape[1]):
            if i == 0 or i == rectified_map.shape[0] - 1 or j == 0 or j == rectified_map.shape[1] - 1:
                # Border cells are set to unexplored (gray)
                rectified_map[i, j] = 128

    # Display the input image, binary map, and rectified map using Matplotlib
    cv2.imwrite("ogm_rectier/map.pgm", rectified_map)
    self.im = Image.open("ogm_rectier/map.pgm")
    self.show_image()


class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.num_neurons = pattern_size[0] * pattern_size[1]
        self.weights = np.zeros((self.num_neurons, self.num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern_flattened = pattern.flatten()
            weight_update = np.outer(pattern_flattened, pattern_flattened)
            np.fill_diagonal(weight_update, 0)  # No self-connections
            self.weights += weight_update

    def update(self, pattern, max_iterations=100, tolerance=1e-6):
        """
        Update the pattern until convergence or reaching the maximum number of iterations.
        """
        old_pattern = pattern.copy().flatten()  # Flatten and copy the input pattern
        if np.all(old_pattern == 0) or np.all(old_pattern == 1):
            # If the input pattern is entirely black or white, return it without updating
            return pattern

        for i in range(max_iterations):
            new_pattern = np.sign(np.dot(self.weights, old_pattern))  # Use the old_pattern for update
            new_pattern = new_pattern.reshape(self.pattern_size)  # Reshape the updated pattern
            if np.allclose(new_pattern, old_pattern.reshape(self.pattern_size), atol=tolerance):
                print(f"Convergence reached at iteration {i + 1}")
                break  # Convergence reached
            old_pattern = new_pattern.flatten()  # Flatten the updated pattern for next iteration
            print(f"Iteration {i + 1}: Pattern changed")
        else:
            print(f"Maximum iterations ({max_iterations}) reached without convergence.")
        return new_pattern

if __name__ == "__main__":
    click_number = 0
    app = MapClass("map.pgm")
    app.mainloop()