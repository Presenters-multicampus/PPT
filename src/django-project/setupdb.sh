
python3 -m venv env
. env/bin/activate

pip install django requests Pillow opencv-python

pip list
echo -e "\npackages installed\n"

echo -e "\ncreate database..."
python manage.py migrate
python manage.py makemigrations snapshot3
python manage.py migrate snapshot3