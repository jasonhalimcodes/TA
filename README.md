# TA [Tugas Akhir]
A Capstone Final Project by Jason Halim C11190014 [Pose and Position Detection with YOLOv4-tiny & OpenCV DNN Library]
# ------------------------------------------------------------------------------
# User - Guide
## Prerequisites
1. Install _dependencies_ OpenCV pada Jetson
```
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install build-essential cmake unzip pkg-config
$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo apt-get install libv4l-dev libxvidcore-dev libx264-dev
$ sudo apt-get install libgtk-3-dev
$ sudo apt-get install libatlas-base-dev gfortran
$ sudo apt-get install python3-dev
```
2. Persiapan _memory space_ untuk melakukan instalasi OpenCV pada Jetson (khususnya Jetson Nano)
```
$ sudo apt-get install nano
$ sudo apt-get install dphys-swapfile
$ sudo nano /sbin/dphys-swapfile
$ sudo nano /etc/dphys-swapfile
```
![image](https://github.com/jasonhalimcodes/TA/assets/116324181/db3f9db6-170f-44d5-bf00-cfdc17d98114)
![image](https://github.com/jasonhalimcodes/TA/assets/116324181/74ae9925-ba33-41dd-9a1c-31be57cb260e)
```
$ wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/blob/main/OpenCV-4-6-0.sh
$ sudo chmod 755 ./OpenCV-4-6-0.sh
$ ./OpenCV-4-6-0.sh
$ rm OpenCV-4-6-0.sh
```

## Setup
1. Hubungkan Jetson dengan kamera web pada usb port
2. Hubungkan Jetson dengan jaringan internet melalui dongle wifi atau ethernet

## Persiapan Deteksi Posisi
1. 

## Menggunakan Sistem Deteksi Pose dan Posisi
1. Buka terminal pada Jetson
2. Jalankan program deteksi pose dan posisi dengan _command_:
```
python3 livRoom.py
```
atau
```
python3 dinRoom.py
```
> dinRoom.py dan livRoom.py merupakan program yang memiliki fungsi yang sama, hanya parameter transformasi homografiknya saja yang berbeda (topL, botL, topR, botR)
