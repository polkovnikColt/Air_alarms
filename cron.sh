sudo apt update
sudo apt install cron
sudo systemctl enable cron 
sudo crontab -e
#0 * * * * /usr/bin/python3 ${script_path}/predict.py >> ./file.log
