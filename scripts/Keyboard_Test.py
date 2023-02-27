from pynput.keyboard import Key, Listener
import time

class Keyboard():
    def __init__(self):
        self.down_keys = []

    def on_press(self, key):
        if not key in self.down_keys:
            self.down_keys.append(key)

    def on_release(self, key):
        if key in self.down_keys:
            self.down_keys.remove(key)

# Collect events until released
keyboard = Keyboard()


with Listener(
        on_press=keyboard.on_press,
        on_release=keyboard.on_release) as listener:
    while True:
        print(keyboard.down_keys)
        time.sleep(0.1)
    listener.join()

