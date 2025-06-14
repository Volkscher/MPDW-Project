## This will have the menu and the controller logic

#### Menus ####

import threading
import time
from project_final_helper import setup_phase2

def start_menu():
    print("MPDW Project Final Phase")
    print("Made by: ")
    print("Pedro Peralta")
    print("Rafael Pires")
    print("Rodrigo Maravilhas")

    print("\n Please choose one of the following options:")
    print("1 - Phase 2")
    print("2 - Phase 3 ()")
    print("3 - Exit")

    valid_inputs = ["1", "2", "3"]
    
    is_input_valid = False

    while(not is_input_valid):
        choice = input("insert option: ")

        if(choice in valid_inputs):
            if(choice == "1"):
                print("choice 1")
            if(choice == "2"):
                print("choise 2")
                phase_2_menu()
            if(choice == "3"):
                print("Exiting...")
                break
        else:
            print("invalid choice")

# Loading animation (runs in a separate thread)
def loading_animation(stop_event):
    dots = 0
    while not stop_event.is_set():
        print("\rLoading" + "." * dots + " " * (3 - dots), end="")
        dots = (dots + 1) % 4  # Cycle: 0 → 1 → 2 → 3 → 0...
        time.sleep(0.5)
    print("\rLoading complete!     ")



# Phase 2 Menu
def phase_2_menu():
    # Main
    stop_event = threading.Event()
    loader_thread = threading.Thread(target=loading_animation, args=(stop_event,))
    loader_thread.start() # starts loading function

    # Run setup
    setup_phase2()
    print("stopping load animation...")
    # Stop loading animation
    stop_event.set()
    loader_thread.join()



def start():
    # We might need to copy the start menu logic here, move it a level up so we can go back and forward
    start_menu()

start()