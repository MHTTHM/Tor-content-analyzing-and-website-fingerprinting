import sys
sys.getdefaultencoding()
import pickle
import numpy as np
from models import HGAT
np.set_printoptions(threshold=1000000000000000)
path = 'D:\\Data for_ Improving Named Entity Recognition in Noisy User-generated Text with Local Distance Neighbor Feature\\HGAT-master\\model\\code\\model\\test_v3.pkl'
file = open(path,'rb')
inf = pickle.load(file,encoding='latin1')       #读取pkl文件的内容
print(inf)
#fr.close()
inf=str(inf)
obj_path = 'F:\\毕设\\python爬虫\\已爬txt\\test_v3_pkl.txt'
ft = open(obj_path, 'w')
ft.write(inf)