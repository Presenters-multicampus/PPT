nohup $(which redis-server) &
nohup $(which celery) -A core worker -l info &

echo -e "\n 완료! 'runserver2.sh' 를 sudo 로 실행하세요. \n"
