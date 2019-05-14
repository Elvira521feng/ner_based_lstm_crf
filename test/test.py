from fastNLP.io.dataset_loader import Conll2003Loader

file_path = '../data/raw_data/test.txt'
loader = Conll2003Loader()
data = loader.load(file_path)

print(type(data))

print(data[0]['tokens'])

print(len(data))
