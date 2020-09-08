from desmume.emulator import DeSmuME
from client.trainer import Trainer
import sys
from contextlib import redirect_stdout

def main(emu):
    emu.open('assets/0168 - Mario Kart DS (U)(SCZ).nds') # a
    win = emu.create_sdl_window()
    emu.savestate.load_file('assets/lol.ds2')
    emu.input.keypad_add_key(64)
    emu.volume_set(0)
    reward_function_addr = int('0x02354B1C', 16)
    #while True:
    #    emu.cycle()
    #    win.draw()
    #    win.process_input()
    #    print(emu.memory.read(reward_function_addr, reward_function_addr, 2, signed=True))

    with open('assets/file.txt', 'w') as f:
        with redirect_stdout(f):
            trainer = Trainer(emu, win, int('0x02354B1C', 16), f)
            trainer.train()


if __name__ == '__main__':
    emulator = DeSmuME()
    main(emulator)