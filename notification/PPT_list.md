# PPT_list

- good e.g.
  - https://www.youtube.com/watch?v=Z_5QJHfJbq8&list=PLtIY1Yjgq7i0BVXinD3fK0DPLmP4mbXA_ : 경진대회 list
  - https://www.youtube.com/watch?v=XBlTqyoDb5s&list=PLFjaMGs3QAviMELcGTvrf6k4CZKJLG18m
  - https://www.youtube.com/watch?v=YO5-By0plqk
  - https://www.youtube.com/watch?v=YivQYeI0vys
  - https://www.youtube.com/watch?v=ck5vVU8qQWA



- bad e.g.

  -  https://www.youtube.com/watch?v=S5c1susCPAE 

    

- PPT algorithms
  - eye detecting : snapshot에서 눈이 몇번 등장하는지 확인
  - hand moving : 어깨, 팔꿈치, 손의 변동을 분석하여 너무 적으면 경직된 발표일 수 있음을 조언
  - face recognition : 감정 표현의 빈도를 분석하여 positive한 표정이 많은지 negative한 표정이 많은지 분석!
  - standing straightly : 허리를 잘 곧추세우고 발표하는지, 자세가 너무 굽지는 않았는지 평가
  - left-right balance : 좌-우 어깨가 한쪽으로 기울지는 않았는지, 양쪽이 대칭되게 서 있어야 함
  - screened by face : 얼굴로 presentation 화면을 가리지는 않았는지
  - too much moving : 이동한 것이 아니라, 몸을 너무 떨거나 바들대지는 않는지 : 각 좌표가 실제로 이동하는지, 진동하는지 확인요
  - talk to ground : 땅바닥과 얘기하는 것은 아닌지
  - moving too fast : 너무 빠르게 움직이는 것은 아닌지
  - hands in the pocket : 주머니에 손을 넣은 것은 아닌지
  - cross arms : 팔짱을 낀 것은 아닌지
  - pointing somewhere by finger : 손가락질 하는 것은 아닌지
  - putting hands on the head : 머리 위에 손을 얹는 것은 아닌지
  - staring at the specific point too much : 같은 곳만 계속 응시하는 것은 아닌지

-     BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

10.142.0.2

- turn on command in PUTTY
  - sudo $(which python) manage.py runserver 10.142.0.2:80 &
  - sudo $(which celery) -A core worker -l info



- file move by using txt file or pickle
  - 템플릿과 view를 봐라



- emotion model 교체
- 매 사진 snapshot마다 



- 오늘 할것 : 알고리즘 전체 구현
  - 구현할 때에 : 각 페이지에 보여줄 알고리즘 / 전체 total score에서 보여줄 알고리즘을 따로
  - 따로 함수로 구현
- 모델 변경 : CNN => resnet (emotion) / face recognition : harr => DNN : ok
- 적용 - deploy
- workflow에 위반되게 했으므로 : git remote add downstream https://github.com/gtpgg1013/PPT.git
- 추가하고 거기에다가 추가 저장하는 식으로 하자.