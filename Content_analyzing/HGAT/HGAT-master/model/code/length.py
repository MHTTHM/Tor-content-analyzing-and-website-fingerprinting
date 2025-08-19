import pickle

def readpkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(data)
    return len(data)

x = ['example', 'test', 'test_v2', 'test_v3','test_v3_ill']

for i in x:
    path = r'D:\博士研究\零碎事情\带学生\朱怡霖\本科毕设\HGAT半监督分类\Data for_ Improving Named Entity Recognition in Noisy User-generated Text with Local Distance Neighbor Feature\HGAT-master\data\{i}\{i}.txt'.format(i=i)
    lines = open(path, 'r', encoding='utf-8').readlines()
    length = len(lines)
    print(f'{i} length: {length}')

for i in x:
    path = f'embeddings/{i}.emb'
    lines = open(path, 'r', encoding='utf-8').readlines()
    length = len(lines)
    print(f'{i} length: {length}')