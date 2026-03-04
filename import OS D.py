import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model langsung
model = pickle.load(open(r"D:\Semester 6\Kontrol Cerdas\Praktikum 3\model.p", "rb"))

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

labels_dict = {0: 'TUTUP', 1: 'BUKA'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmark,
                mp_hands.HAND_CONNECTIONS
            )

            for i in range(len(hand_landmark.landmark)):
                x = hand_landmark.landmark[i].x
                y = hand_landmark.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmark.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

            prediction = model.predict(np.asarray(data_aux).reshape(1, -1))
            predicted_character = labels_dict[int(prediction[0])]

            cv2.putText(frame, predicted_character, (200, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()