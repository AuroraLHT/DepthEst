#!/bin/bash
dont=(velodyne_points
    image_00
    image_01
    )

zipfiles="$( ls | grep .zip )"
echo "${zipfiles}"

for zf in ${zipfiles}; do
    unzip $zf
    where=${zf:0:10}'/'${zf:0:-4}
    echo "${where}"
    for d in ${dont[@]}; do
        rm -r $where'/'$d
    done
done
  
