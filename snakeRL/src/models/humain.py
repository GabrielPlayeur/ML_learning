
class Humain:
    def __init__(self):
        self.key_to_action = {"z": "u", "q": "l", "s": "d", "d": "r"}

    def action(self, state):
        key = None
        while not self.valide_key(key) :
            key = input("")
        return self.key_to_action[key]

    def valide_key(self, key: str):
        return key in self.key_to_action.keys()