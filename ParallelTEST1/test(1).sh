#test.sh
# !/bin/sh
pssh -h $PBS_NODEFILE -i "if [ ! -d \"/home/s2013540\" ];then mkdir -p \"/home/s2013540\"; fi" 1>&2  
pscp -h $PBS_NODEFILE /home/s2013540/main /home/s2013540 1>&2  
/home/s2013540/main  
perf stat -e instructions,cycles -r 100 /home/s2013540/main
