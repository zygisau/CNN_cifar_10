import csv


class CSVWriter:
    def __init__(self, file_name):
        self.file_name = file_name
        csv_file = open(file_name, 'w', newline='', encoding="utf-8")
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        self.reader = writer

    def append_to_file(self, new_line):
        self.reader.writerow(new_line)
