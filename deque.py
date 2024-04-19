from collections import deque

class LimitedDict:
    """This is a custom deque class that will allow the truncation of older historical rounds for each arm"""
    def __init__(self, max_values):
        self.max_values = max_values
        self.dict = {}

    def add_value(self, key, value):
        if len(self.get_values(key)) > self.max_values:
            self.dict[key].pop(0)
            self.dict[key].append(value)
        else:
            self.dict[key].append(value)

    def set_new_max(self, key, new_max):
        if key in self.dict:
            old_values = list(self.dict[key])
            self.dict[key] = deque(old_values, maxlen=new_max)
            self.max_values = new_max
        else:
            raise KeyError(f"Key '{key}' not found in the LimitedDict.")

    def add_key(self, key):
        if key not in self.dict:
            self.dict[key] = []

    def get_values(self, key):
        return list(self.dict.get(key, []))