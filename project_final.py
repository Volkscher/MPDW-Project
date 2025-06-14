## This will have the menu and the controller logic

#### Menus ####

import threading
import time

from matplotlib import pyplot as plt
from project_final_helper import display_comparison, search_caption_by_image, search_frame_by_image, setup_phase2, search_frame_by_text
from project_final_phase3_helper import phase_3_mega_function
from PIL import Image

def run_with_loading(task_func, *args, message="Loading", delay=0.5, **kwargs):
    stop_event = threading.Event()

    def loading_animation():
        dots = 0
        while not stop_event.is_set():
            print("\r" + message + "." * dots + " " * (3 - dots), end="")
            dots = (dots + 1) % 4
            time.sleep(delay)
        print("\r" + message + " complete!    ")

    loader_thread = threading.Thread(target=loading_animation)
    loader_thread.start()

    result = task_func(*args, **kwargs)  # Run your actual task

    stop_event.set()
    loader_thread.join()

    return result

def start_menu():
    print("MPDW Project Final Phase")
    print("Made by: ")
    print("Pedro Peralta")
    print("Rafael Pires")
    print("Rodrigo Maravilhas")
    
    is_input_valid = False

    while(not is_input_valid):
        print("\nPlease choose one of the following options:")
        print("1 - Phase 2 Indexing & Querying with CLIP")
        print("2 - Phase 3 Prompt Based Image Generation & Visual Question Evaluation")
        print("3 - Exit")

        valid_inputs = ["1", "2", "3"]
        choice = input("insert option: ")

        if(choice in valid_inputs):
            if(choice == "1"):
                phase_2_menu()
            if(choice == "2"):
                phase_3_menu()
            if(choice == "3"):
                print("Exiting...")
                is_input_valid = True
                break
        else:
            print("invalid choice")

# Phase 2 Menu
def phase_2_menu():
    # Main
    run_with_loading(setup_phase2)
    print("Videos Downloaded and Indexed. Querying is now possible")

    is_input_valid = False

    while(not is_input_valid):
        print("Options:")
        print("1 - Text to Image/Frame Retrieval")
        print("2 - Image to Text Retrieval")
        print("3 - Image to Image Retrieval")
        print("4 - Back")

        valid_inputs = ["1", "2", "3", "4"]

        choice = input("insert option: ")

        if(choice == "1"):
            print("Text to Frame Retrieval Selected")
            prompt = input("insert your text query: ")
            results = run_with_loading(search_frame_by_text, prompt, message= "Searching")
            print("Frames Retrieved. Displaying Now...")
            
            for hit in results:
                print("printing results...")
                
                img = Image.open(f"keyframes/{hit['video_id']}/{hit['frame_path']}")
                plt.imshow(img)
                plt.title(f"Score: {hit['score']:.4f}", fontsize=12)
                # Display caption below the image
                plt.text(
                    0, img.height + 10,              # X, Y coordinates
                    "\n".join(hit['captions']),             # if `captions` is a list
                    fontsize=10,
                    va='top'
                )
                plt.xticks([])
                plt.yticks([])

                print("To view next hit, close the current plot!")

                plt.show()

            is_input_valid = True        

        if(choice == "2"):
            print("Image to Text Retrieval Selected")
            print("Please copy images you wish to query to the directory '/sample_images'")
            filename = input("insert the filename for the image (including extension): ")

            results = run_with_loading(search_caption_by_image, "./sample_images/" + filename, 5, message= "Searching")
            
            for hit in results:
                print("printing results...")
                
                img = Image.open(f"keyframes/{hit['video_id']}/{hit['frame_path']}")
                plt.imshow(img)
                plt.title(f"Score: {hit['score']:.4f}", fontsize=12)
                # Display captions below the image
                plt.text(
                    0, img.height + 10,              
                    "\n".join(hit['captions']),             
                    fontsize=10,
                    va='top'
                )
                plt.xticks([])
                plt.yticks([])

                print("To view next hit, close the current plot!")

                plt.show()

            is_input_valid = True

        if(choice == "3"):
            print("choice 3")
            print("Image to Image Retrieval Selected")
            print("Please copy images you wish to query to the directory '/sample_images'")
            filename = input("insert the filename for the image (including extension): ")

            results = run_with_loading(search_frame_by_image, "./sample_images/" + filename, 5, message= "Searching")
            
            for hit in results:
                print("printing results...")
                print("To view next hit, close the current plot!")
                display_comparison(filename, hit)

            is_input_valid = True
        
        if(choice == "4"):
            is_input_valid = True



def phase_3_menu():
    print("Phase 3 - Prompt Based Image Generation and Visual Question Evaluation")

    user_input = input("insert a prompt to generate an image: ")
    phase_3_mega_function(prompt= user_input)
    # Run evaluation for each image returned by CLIP




def start():
    # We might need to copy the start menu logic here, move it a level up so we can go back and forward
    start_menu()

start()