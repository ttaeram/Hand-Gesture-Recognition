import cv2
import mediapipe as mp
import numpy as np
import torch
import random
from model.cnn_lstm import CNN_LSTM

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(0)

if device.type == 'cuda':
    torch.cuda.manual_seed_all(0)
    print('cuda ready')
else:
    print('running on cpu')

CNN_LSTM_model = CNN_LSTM(
    input_size=99,
    output_size=128,
    hidden_size=64,
    num_classes=30
).to(device)

state_dict = torch.load(
    "./hmi_FGCS.pt",
    map_location=device,
    weights_only=True
)
CNN_LSTM_model.load_state_dict(state_dict)
CNN_LSTM_model.eval()

gesture = {'Select Drone':0, 'Select Group':1, 'Select Mode':2, 'ARM':3, 'DISARM':4, 'TAKEOFF':5, 'LAND':6, 'RTL':7,
           'Change Altitude':8, 'Change Speed':9, 'Move Up':10, 'Move Down':11, 'Rotate CW':12, 'Move Forward':13,
           'Move Backward':14, 'Move Right':15, 'Move Left':16, 'Rotate CCW':17, 'Cancel':18, 'Check':19,
           'One':20, 'Two':21, 'Three':22, 'Four':23, 'Five':24, 'Six':25, 'Seven':26, 'Eight':27, 'Nine':28, 'Ten':29}

actions = ['Select Drone', 'Select Group', 'Select Mode', 'ARM', 'DISARM', 'TAKEOFF', 'LAND', 'RTL', 'Change Altitude',
           'Change Speed', 'Move Up', 'Move Down', 'Rotate CW', 'Move Forward', 'Move Backward', 'Move Right', 'Move Left',
           'Rotate CCW', 'Cancel', 'Check', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']

seq_length = 30
seq = []
action_seq = []
thres_fr = 30

cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FPS, 30)
print("설정된 FPS : ", cap.get(cv2.CAP_PROP_FPS))

win_name = "HMI Gesture"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 800, 600)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)

    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum(
                'nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
            ))
            angle = np.degrees(angle)

            gesture_joint = np.concatenate([joint.flatten(), angle]).astype(np.float32)
            seq.append(gesture_joint)

            mp_drawing.draw_landmarks(
                img,
                res,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            input_data = torch.tensor(input_data, dtype=torch.float32, device=device)

            with torch.no_grad():
                y_pred = CNN_LSTM_model(input_data)
                values, indices = torch.max(y_pred, dim=1)

            model_confidence = float(values.item())

            if model_confidence < 0.9:
                cv2.putText(img, f"Low conf: {model_confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                continue

            pred_idx = int(indices.item())
            action = actions[pred_idx]
            action_seq.append(action)

            if len(action_seq) < thres_fr:
                continue

            if all(a == action_seq[-1] for a in action_seq[-thres_fr:]):
                result_action = action_seq[-1]
                print(f"Result : {result_action}")
                print(f"conf: {model_confidence:.2f}")
                action_seq = []

    cv2.imshow(win_name, img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()