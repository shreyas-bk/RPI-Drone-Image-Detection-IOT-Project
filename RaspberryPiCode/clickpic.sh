#!/bin/bash
#The above line tells the system to execute the script using the Bash shell.

rm -rfv /home/pi/Documents/download/Upload/*

a=0
while [ $a -lt 11 ]
do   
    DATE=$(date +"%Y-%m-%d_%H%M%S") 
    echo $DATE
    raspistill -o /home/pi/Documents/download/zipthis/$DATE.jpg
    python3 gps.py $DATE
    a=`expr $a + 1`
    sleep 4

done

DATE2=$(date +"%Y-%m-%d_%H%M%S") 

if [ -z "$(ls -A /home/pi/Documents/download/zipthis)" ]; then
   echo "Empty"
else
   zip -r /home/pi/Documents/download/Upload/$DATE2.zip  /home/pi/Documents/download/zipthis
fi

rm -rfv /home/pi/Documents/download/zipthis/*

rclone sync -v /home/pi/Documents/download drive:images
