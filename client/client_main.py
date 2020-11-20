from desmume.emulator import DeSmuME
from teacher import Teacher
import socket
import numpy as np
import threading
import time
from contextlib import redirect_stdout

serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
emu = DeSmuME()
inference = None
frame = 0
do_training = False
do_inference = False

def main():
    emu.open('assets/0168 - Mario Kart DS (U)(SCZ).nds')
    win = emu.create_sdl_window()
    emu.NB_STATES = 100
    emu.savestate.load_file('assets/yoshi.dst')

    emu.volume_set(0)

    # if not do_training:
    #     global frame
    #     while frame := frame+1:
    #         #time.sleep(.01)
    #         emu.cycle()
    #         win.draw()
    #         print(emu.memory.read(int('0x0234CCFC', 16),int('0x0234CCFC', 16),2,signed=True))
    #         if not inference:
    #             emu.input.keypad_update(0)
    #         if not do_training:
    #             win.process_input()
    #         if frame==1000:
    #            cheat(emu)
    #            print('chaeated')

    print(emu.input.keypad_get())
    # ugly 0x0233EF5C
    # peach gardens time trial 0235EB1C
    # shroom ridge time trial 023599DC
    # yoshi falls 0x0234CCFC

    with open('assets/file2.txt', 'w') as logfile, open('training_data3/labels.txt','a') as labelfile:
        with redirect_stdout(logfile):
            teacher = Teacher(emu, win, int('0x0234CCFC', 16), logfile, labelfile, headless=False)
            while True:
                teacher.train()
                emu.savestate.load_file('assets/yoshi.dst')

def client_inference():
    global inference
    serv.connect(('127.0.0.1', 65432))
    client_send_screen()
    while True:
        client_send_screen()
        raw = serv.recv(57).decode()
        t = raw[1:-1].split(',')
        inference = (int(t[0]), int(t[1]))
        time.sleep(.0001)

def control():
    while True:
        # while frame%20 != 0:
        #     pass
        print(inference)
        if inference:
            emu.input.keypad_rm_key(inference[1])
            emu.input.keypad_update(inference[1]+1)
        time.sleep(.01)


def client_send_screen():
    img = np.asarray(emu.screenshot())
    serv.send(img.tobytes())

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
    if not do_training and do_inference:
        t1 = threading.Thread(target=client_inference)
        t1.start()
        t2 = threading.Thread(target=control)
        t2.start()
    main()