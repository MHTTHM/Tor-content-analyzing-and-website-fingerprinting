from os.path import join, abspath, dirname, pardir
BASE_DIR = abspath(join(dirname(__file__), pardir))
output_dir = join(BASE_DIR, 'RF/dataset/')
split_mark = '\t'
OPEN_WORLD = False
MONITORED_SITE_NUM = 182
MONITORED_INST_NUM = 100
UNMONITORED_SITE_NUM = 30000
UNMONITORED_SITE_TRAINING = 1000
model_path = 'pretrained/'

#num_classes = 101

if OPEN_WORLD:
    num_classes = MONITORED_SITE_NUM+1
else:
    num_classes = MONITORED_SITE_NUM

num_classes_ow = 1
# Length of TAM
max_matrix_len = 100
# Maximum Load Time
maximum_load_time = 45

max_trace_length = 5000
