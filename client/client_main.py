from desmume.emulator import DeSmuME
from teacher import Teacher
from app import App
import socket
import numpy as np
import threading
import time
from contextlib import redirect_stdout

label_classes = {
    '(30, 32)': 0,
    '(25, 32)': 1,
    '(20, 32)': 2,
    '(15, 32)': 3,
    '(10, 32)': 4,
    '(05, 32)': 5,
    '(00, 00)': 6,
    '(05, 16)': 7,
    '(10, 16)': 8,
    '(15, 16)': 9,
    '(20, 16)': 10,
    '(25, 16)': 11,
    '(30, 16)': 12
}

inv_labels = {label_classes[x]: x for x in label_classes}

serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
emu = DeSmuME()
inference = None
preds = None
frame = 0
do_training = False
do_inference = False

def main():
    emu.open('assets/0168 - Mario Kart DS (U)(SCZ).nds')
    win = emu.create_sdl_window()
    emu.NB_STATES = 100
    emu.savestate.load_file('assets/mario.dst')

    emu.volume_set(0)

    # if not do_training:
    #     global frame
    #     while frame := frame+1:
    #         #time.sleep(.01)
    #         emu.cycle()
    #         win.draw()
    #         #print(emu.memory.read(int('0x0234CCFC', 16),int('0x0234CCFC', 16),2,signed=True))
    #         app.update_buffer(emu.screenshot(),
    #                           emu.memory.read(int('0x0233EF5C', 16), int('0x0233EF5C', 16), 2, signed=True),
    #                           preds,
    #                           frame)
    #         # if not inference:
    #         #     emu.input.keypad_update(0)
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
    # luigi manion 0235DC7C
    # mario circuit 023688DC
    # bridge 02366D5C

    with open('assets/file2.txt', 'w') as logfile, open('training_data2/labels.txt','a') as labelfile:
        with redirect_stdout(logfile):
            teacher = Teacher(emu, win, int('0x023688DC', 16), logfile, labelfile, headless=False)
            while True:
                teacher.train()
                emu.savestate.load_file('assets/mario.dst')

def client_inference():
    global inference
    global preds
    serv.connect(('127.0.0.1', 65432))
    client_send_screen()
    while True:
        client_send_screen()
        raw = np.fromstring(serv.recv(4096).decode().replace('[','').replace(']',''), dtype=np.float, sep=',')

        preds = raw
        i = inv_labels[np.argmax(raw)]
        t = i[1:-1].split(',')
        inference = (int(t[0]), int(t[1]))
        time.sleep(.01)

def control():
    while True:
        # while frame%20 != 0:
        #     pass
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
    # threading.Thread(target=main).start()
    # if True:
    #     app = App()
    #     app.mainloop()
    main()

