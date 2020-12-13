from matplotlib import pyplot


class ImageSaver:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = 'image_'
        self.file_extension = '.png'
        self.plt = pyplot

    def save_image(self, index):
        self.plt.savefig(self.file_path + self.file_name + str(index) + self.file_extension)
