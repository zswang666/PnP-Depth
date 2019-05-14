import math

class ErrorLogger:
    """A class for error logging"""
    def __init__(self, keys, formats, filename):
        self.errors = {}
        self.formats = formats
        for key in keys:
            self.errors[key] = 0.0
        self.N = 0
        self.filename = filename
        self.keys = keys        
        self.log_header()


    def log_header(self):
        with open(self.filename, 'a') as f:
            for key, frmt, i in zip(self.keys, self.formats, range(len(self.errors))):
                f.write('{{: >{}}}'.format(frmt[0]).format(key))
                if i < len(self.errors)-1:
                    f.write(',')
            f.write('\n')
    def log(self):
        with open(self.filename, 'a') as f:
            for key, frmt, i in zip(self.keys, self.formats, range(len(self.errors))):
                f.write('{{: >{}.{}}}'.format(frmt[0], frmt[1]).format(self.errors[key]/self.N))
                if i < len(self.errors)-1:
                    f.write(',')
            f.write('\n')
    def update_log_string(self, values):
        str = ""
        for key, fmt in zip(self.keys, self.formats):
            str += ('{}: {{: >{}.{}}} ({{: >{}.{}}}) '.
                    format(key, fmt[0], fmt[1], fmt[0], fmt[1]).
                    format(values[key], self.errors[key]/self.N))
        return str
    
    def update(self, values):
        for key in values:
            self.errors[key] += values[key]
        self.N += 1
    def clear(self):
        for key in self.errors:
            self.errors[key] = 0
        self.N = 0
    def get(self, key):
        return self.errors[key]/self.N

            
        
        

