from tqdm import tqdm
from time import sleep

arr = [1, 2, 3, 4, 5]
with tqdm(total=len(arr), desc="", leave=False) as pbar:
    for i in arr:
        pbar.set_description(str(i))
        sleep(1)
        pbar.update(1)