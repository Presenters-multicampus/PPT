
python3 -m venv env
. env/bin/activate
pip install django requests Pillow

echo -e "\ncreate database..."
python manage.py migrate > /dev/null 2>&1
python manage.py makemigrations snapshot2 > /dev/null 2>&1
python manage.py migrate snapshot2 > /dev/null 2>&1