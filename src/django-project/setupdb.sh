
python3 -m venv env
. env/bin/activate

echo -e "\n 필요 패키지 설치중... \n"
pip install django requests Pillow opencv-python
pip install 'celery[redis]'
pip install redis==2.10.6

pip list
echo -e "\n 필요 패키지 설치 완료 \n"

echo -e "\n 데이터베이스 만드는중... \n"
python manage.py migrate
python manage.py makemigrations snapshot3
python manage.py migrate snapshot3

echo -e "\n redis & celery 실행중... \n"
nohup $(which redis-server) &
nohup $(which celery) -A core worker -l info &

echo -e "\n 완료! 'runserver.sh' 를 sudo 로 실행하세요. \n"