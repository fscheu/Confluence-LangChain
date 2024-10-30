###### Funciones de utilidad ######
import configparser


def read_ini(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


config = read_ini("config.ini")
