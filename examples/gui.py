import cv2
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key


ACTION_MAPPING = {
    'w': np.array([-1., 0., 0., 0.]),
    's': np.array([1., 0., 0., 0.]),
    'a': np.array([0., -1., 0., 0.]),
    'd': np.array([0., 1., 0., 0.]),
    Key.space: np.array([0., 0., 1., 0.]),
    Key.shift: np.array([0., 0., -1., 0.]),
    'e': np.array([0., 0., 0., 1.]),
    'q': np.array([0., 0., 0., -1.]),
}


class GUI:
    def __init__(self, height: int = 720, fps: int = 25) -> None:
        self._init_keyboard_listener()
        self.height = height
        self.wait_time = int(1000 / fps)

    def _init_keyboard_listener(self) -> None:
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self._keys_pressed = []
    
    def _on_press(self, key) -> None:
        if key not in self._keys_pressed:
            self._keys_pressed.append(key)

    def _on_release(self, key) -> None:
        if key in self._keys_pressed:
            self._keys_pressed.remove(key)

    def display(self, image: np.array) -> None:
        resized = cv2.resize(image[..., ::-1], (int(self.height * image.shape[1] / image.shape[0]), self.height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Viewer", resized)
        cv2.waitKey(self.wait_time)

    def get_action(self, action_scale: float = 0.2) -> np.array:
        action = np.zeros(4)
        for key in self._keys_pressed:
            try:
                action += action_scale * ACTION_MAPPING[key.char]
            except AttributeError:
                try:
                    action += action_scale * ACTION_MAPPING[key]
                except KeyError:
                    pass
        return action
