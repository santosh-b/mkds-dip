from desmume.emulator import DeSmuME
from teacher import Teacher
import socket
from contextlib import redirect_stdout

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def main(emu):
    #client.connect('127.0.0.1', 65432)
    emu.open('assets/0168 - Mario Kart DS (U)(SCZ).nds')
    win = emu.create_sdl_window()
    emu.NB_STATES = 100
    emu.savestate.load_file('assets/lol3.ds2')

    emu.input.keypad_add_key(64)
    emu.volume_set(0)
    reward_function_addr = int('0x02354B1C', 16)
    # frame = 0
    # while frame := frame+1:
    #     emu.cycle()
    #     win.draw()
    #     win.process_input()
    #     if frame==1000:
    #        cheat(emu)
    #        print('chaeated')
    #     print(emu.input.keypad_get())

    # print(emu.input.keypad_get()) #0x0233EF5C
    # # peach gardens time trial 0235EB1C
    # # shroom ridge time trial 023599DC
    #
    with open('assets/file.txt', 'w') as logfile, open('training_data/labels.txt','a') as labelfile:
        with redirect_stdout(logfile):
            teacher = Teacher(emu, win, int('0x0233EF5C', 16), logfile, labelfile, headless=True)
            while True:
                teacher.train()
                emu.savestate.load_file('assets/lol3.ds2')

#def client_send_img(img):


def cheat(emu):
    '''
    miniature action replay cheat interpreter
    loads the following action replay cheat
    92056E14 00004002
    12056E12 000020FF
    D2000000 00000000
    '''
    emu.memory.write_long(int('0x02056E14', 16), int('0x4002', 16))
    emu.memory.write_long(int('0x02056E12', 16), int('0x20FF',16))



if __name__ == '__main__':
    emulator = DeSmuME()
    main(emulator)