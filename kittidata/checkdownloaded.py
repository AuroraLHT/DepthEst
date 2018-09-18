from pathlib import Path
base = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"

with open('fullurl.txt','r') as f:
    ls = set( l.rstrip() for l in f) 

zips = set( z.name for z in  Path('.').glob('*.zip'))
checkpoints = set(cp.stem for cp in Path('.').glob('*.ari*'))

noneededzips = zips - checkpoints

noneededzips = set(base+z for z in noneededzips) | set(base+z[:-9]+'/'+z for z in noneededzips)

ls = ls - noneededzips

with open('resturl.txt', 'w') as f:
    f.write("\n".join(tuple(ls)))
