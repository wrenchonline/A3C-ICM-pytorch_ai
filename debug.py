from MAMEToolkit.emulator import run_cheat_debugger
import os
# os.system("Xvfb :0 -screen 0 800x600x16 +extension RANDR &")
# os.environ["DISPLAY"] = ":0"

roms_path = "/home/wrench/roms" # Replace this with the path to your ROMs
game_id = "sfiii3n"
run_cheat_debugger(roms_path, game_id)