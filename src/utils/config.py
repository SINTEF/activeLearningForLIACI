# tmp_dir = 'assets/tmp/'
tmp_dir = '../tmp/'
bench_dir = '../benchmarks/'
curr_bench = '2022_12_07_1709'
model_path = '../benchmarks/' + curr_bench + '/epochs-30_batchsize-50_lr-0.0004_vsplit-0.1_v2-False/model/'
new_images_dir = '../user_annotations/'

threshold = 0.35
fraction = 0.3
wiggle_room = 0.06
n_samples = 100
n_labels = 9

mc_step = 1 # step rate for mc dropout, turn up if mc takes too much time

# Comment in the one you want to use
unc_type = 'MC' # Monte Carlo dropout
# unc_type = 'TI' # Threshold interval

n_mem_frames = 2500 # Set this lower if HW is bad