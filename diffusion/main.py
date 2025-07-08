from preprocessor import create_dataset
from pathlib import Path
from tqdm import tqdm

root_data_dir = Path("D://insat_data_bah2025//Order//Jul25_133821")
save_dir = Path('TIR1Data')

count = 0
for i in tqdm(list(root_data_dir.glob('*.h5'))):
    create_dataset(i,save_dir,str(count))
    count +=1


print(f' >> >> >> {(len(list(save_dir.glob('*'))))} files successfully saved at {save_dir.name} << << <<\n')
    