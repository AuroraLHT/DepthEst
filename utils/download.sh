mkdir ~/KITTI
mkdir ~/KITTI/rawdata
wget -i kitti_download.txt -P ~/KITTI/rawdata/

ln -s ~/KITTI/rawdata ./data
