import pickle

class PickleFileOperator():
    def __init__(self, data=None, file_path=None):
        self.__data = data
        self.__file_path = file_path

    def save(self):
        with open(self.__file_path, 'wb') as f:
            pickle.dump(self.__data, f)

    def read(self):
        with open(self.__file_path, 'rb') as f:
            data = pickle.load(f)
        return data


def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        
def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data