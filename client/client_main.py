from desmume.emulator import DeSmuME

def main(emu):
    emu.open('assets/0168 - Mario Kart DS (U)(SCZ).nds') # a
    win = emu.create_sdl_window()

    frame = 0
    while frame := frame+1:
        emu.cycle()
        win.draw()
        win.process_input()


if __name__ == '__main__':
    emulator = DeSmuME()
    main(emulator)