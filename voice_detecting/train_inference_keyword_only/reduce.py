# Nam, Hyun Joon, 2020 10/19
# python reduce.py ..\data\2020_norm_melsp ..\data\reduced_2020_melsp 50
import os, sys, shutil, random, tqdm
src_dir = sys.argv[1];  dst_dir = sys.argv[2]; ratio=float(sys.argv[3])/100.;
assert(ratio > 0.1),'Ratio {} too small.'.format(ratio*100)
os.makedirs(dst_dir, exist_ok=True);

for set_dir in os.listdir(src_dir):
    if not os.path.isdir(os.path.join(src_dir,set_dir)): 
        shutil.copy2(os.path.join(src_dir,set_dir),os.path.join(dst_dir,set_dir))
    else:
        os.makedirs(os.path.join(dst_dir,set_dir), exist_ok=True);
        print(set_dir)
        for file_ in tqdm.tqdm(os.listdir(os.path.join(src_dir,set_dir))):
            if 'UNKNOWN' in file_: continue
            rand_val=random.random();
            if rand_val < ratio:
                shutil.copy2(os.path.join(src_dir,set_dir,file_),os.path.join(dst_dir,set_dir,file_))
