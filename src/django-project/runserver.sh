
. env/bin/activate
echo -e "\nif you want to stop server,\n\n\t'ps -ef | grep runserver'\n\n\t'sudo kill -9 <PID>'\n"
nohup $(which python) manage.py runserver 172.31.43.89:80 &