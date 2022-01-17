
import os
import shutil
import sys

os.makedirs('/data2/johan/biodata2/TrainSet2/BF', exist_ok=True)
os.makedirs('/data2/johan/biodata2/TrainSet2/SHG', exist_ok=True)
os.makedirs('/data2/johan/biodata2/ValidationSet1/BF', exist_ok=True)
os.makedirs('/data2/johan/biodata2/ValidationSet1/SHG', exist_ok=True)

os.makedirs('/data2/johan/biodata2/TestSet/BF', exist_ok=True)
os.makedirs('/data2/johan/biodata2/TestSet/SHG', exist_ok=True)

#sys.exit(0)
#out_dir = '/data2/johan/biodata2/TrainSet2/'
#dr = '/data2/johan/biodata/TrainSet'

#out_dir = '/data2/johan/biodata2/ValidationSet1/'
#dr = '/data2/johan/biodata/Validation1Set'

out_dir = '/data2/johan/biodata2/TestSet/'
dr = '/data2/johan/biodata/TestSet'

for file in os.listdir(dr):
    pth_in = os.path.join(dr, file)
    file_pth = file
    #file_pth = os.path.join(out_dir, file)
    #if '/T_' in pth and 'BF' in pth:
    #    pass
    if 'BF' in file_pth:
        #pth = pth.string.replace('/R_', '/')
        file_pth = file_pth.replace('_BF', '')
        file_pth = 'BF/' + file_pth
    #elif '/T_' in pth and 'SHG' in pth:
    #    pass
        #pth = pth.string.replace('/T_', '/')
    elif 'SHG' in file_pth:
        #pth = pth.string.replace('/R_', '/')
        file_pth = file_pth.replace('_SHG', '')
        file_pth = 'SHG/' + file_pth
    else:
        continue
    pth = os.path.join(out_dir, file_pth)
    print(pth)
    shutil.copy2(pth_in, pth)


    
