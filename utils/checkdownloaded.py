from pathlib import Path
base = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"

with open('fullurl.txt','r') as f:
    ls = [l.rstrip() for l in f] 

zips = tuple(Path('.').glob('*.zip'))
checkpoints = tuple(Path('.').glob('*.ari*'))
checkpointnames = tuple( checkpoint.stem for checkpoint in checkpoints)

for zipfile in zips:
    if zipfile.name not in checkpointnames:
        if "drive" not in zipfile.name:
            rms=base + zipfile.name
        else:
            rms=base+zipfile.stem[:-5]+'/'+zipfile.name
        try:
            ls.remove(rms)
        except Exception as e:
            print(rms)
            print(e)
with open('resturl.txt', 'w') as f:
    f.write("\n".join(ls))
