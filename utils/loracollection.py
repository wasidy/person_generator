import os


class LoraCollection():
    def __init__(self, base_path):
        self.base_path = base_path
        self.loras = dict()
        categories_folders = os.listdir(self.base_path)

        for folder in categories_folders:
            cat_dir = os.path.join(base_path, folder, '')
            if os.path.isdir(cat_dir):
                cat_key = folder
                self.loras[cat_key] = list()
                cat_type_dir = os.path.join(cat_dir, '')
                for lora_type in os.listdir(cat_type_dir):
                    s = os.path.join(cat_type_dir, lora_type, '')
                    if os.path.isdir(s):
                        lora_files = os.listdir(s)
                        for s2 in lora_files:
                            full_path = s+s2
                            if os.path.isfile(full_path):
                                name, ext = os.path.splitext(s2)
                                if ext == '.safetensors':
                                    show_name = f'{str.upper(cat_key)} - {str.upper(lora_type)} - {name}'
                                    data = list([name, full_path, lora_type, show_name])
                                    self.loras[cat_key].append(data)

    def get_path(self, name):
        for i in self.loras.keys():
            for j in self.loras[i]:
                if j[3] == name:
                    return j[1]
        return None

    def get_trigger_words(self, name):
        for i in self.loras.keys():
            for j in self.loras[i]:
                if j[3] == name:
                    fname = os.path.splitext(j[1])[0]+'.txt'
                    if os.path.exists(fname):
                        try:
                            f = open(fname)
                        except Warning:
                            print('Trigger words not found')
                            words = ''
                        else:
                            with f:
                                words = f.readline()
                    else:
                        words = ''
                    return words

    @property
    def categories(self):
        return list(self.loras.keys())

    @property
    def get_list(self):
        loras = list()
        for i in self.loras.keys():
            for j in self.loras[i]:
                loras.append(j[3])
        return loras

    def get_list_by_key(self, key):
        loras_list = []
        for i in self.loras[key]:
            loras_list.append(i[3])
        return loras_list
        