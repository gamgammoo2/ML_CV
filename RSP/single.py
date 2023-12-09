import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 1 #지금 한손만 해볼거라 1로 설정
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
} #이미 train 해놓은 gesture 데이터. 손가락 관절의 각도와 각각의 라벨
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'} #가위바위보에서는 위의 학습 데이터에서 0이 묵 5가 빠 9가 가위에 해당하므로 우리가 사용할 rps에 대해서 명시해줌

# MediaPipe hands model -손인식
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5, #0.5로 해두는게 제일 좋다.
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',') #genfromtxt : csv에서 데이터를 가져올때 header(파일 첫 줄) 지워주고, 데이터 타입을 실수 값으로 가져온다.
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()#opencv에 knn을 사용해서 학습을 시킴
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0) #opencv의 videocapture을 사용해서 이미지를 읽어온다. 웹캠을 열어줌. 웹캠이 여러개 일 수 있으니까 0,1,2 인지 잘 확인 후 조정.

while cap.isOpened(): #카메라가 열여있으면
    ret, img = cap.read() #한 프레임씩 이미지를 읽어옴. ret은 '성공했다'라는 상태
    if not ret: #성공하지 못한 상태면 계속 하는 것.
        continue

    img = cv2.flip(img, 1) #opencv로 받아온 한 프레임을 이미지 좌우 반전 시켜줌
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # oepncv는 색상이 BGR 기반인데, mdiapipe는 RGB 기반임. 그래서 전환시켜주는 것

    result = hands.process(img) # hands.process() 하면 전처리 및 모델 추론을 함께 실행해줌.

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #이미지를 다시 출력해야하므로 RGB를 BGR로 전환 

    if result.multi_hand_landmarks is not None: #손을 인식했다면(인식하면 result.multi_hand_landmarks 는 true가 될 거임)
        for res in result.multi_hand_landmarks: #여러개의 손을 인식했었을 수 있으므로.
            joint = np.zeros((21, 3)) #손의 joint(총 21개)를 xyz좌표 내에 저장.
            for j, lm in enumerate(res.landmark): #랜드마크는 mediapipe의 joint 숫자부분에 해당
                joint[j] = [lm.x, lm.y, lm.z] #랜드마크의 x,y,z좌표를 joint에 저장

            # Compute angles between joints -각 joint를 가지고 각도 계산
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3] - 벡터계산 - 각 관절에 대한 벡터를 계산
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # np.linalg.norm(v,axis=1) : 각 벡터의 길이 - 길이로 나눠줌으로서 nomalize 함.(단위벡터)

            # Get angle using arcos of dot product -정규화 된 벡터들은 내적에 arccos을 하면 사이 각도가 나온다.
            angle = np.arccos(np.einsum('nt,nt->n', #einsum 연산을 통해 행렬, 벡터의 내적, 외적, 전치, 행렬곱 등을 일관성있게 표현할 수 있다.
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree - 라디안으로 나온 angle을 degree로 바꿔줌

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result -가위바위보에 해당하는 제스쳐만 표시 (만약 다른 제스쳐도 표시하고 싶으면 아래의 othrer gesture을 주석해제 바람)
            if idx in rps_gesture.keys():
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            # Other gestures
            # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) #손가락 마디마디에 landmark를 나타내는 함수

    cv2.imshow('Game', img) #그림을 실제로 보여주는 함수
    if cv2.waitKey(1) == ord('q'):
        break