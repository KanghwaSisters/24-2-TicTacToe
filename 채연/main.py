# -*- coding: utf-8 -*-

from cv2_videocapture import initialize_camera, get_frame
from board_recognition import TicTacToeCNN, extract_board, classify_board, classify_cell
import cv2
import torch
import time

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TicTacToeCNN().to(device)
    model.load_state_dict(torch.load(r"C:\Users\user\Desktop\TicTacToe\model_state_dict.pt", map_location=device, weights_only=True))
    model.eval() 
    
    '''q
    카메라 종류
    0: 노트북 카메라
    1: 폰 웹캠
    '''
    vcap = initialize_camera(0)

    prev_pred = None  # 이전 예측 값 저장
    stable_pred = None  # 안정화된 보드 상태 저장
    change_detected = False  # 보드 상태 변화 여부
    last_change_time = time.time()  # 마지막으로 변화가 감지된 시간
    stability_duration = 2  # 안정화 시간 기준 (초)

    print("Press 'q' to quit the program.")

    while True:
        frame = get_frame(vcap)
        if frame is None:
            break
        frame = cv2.bitwise_not(frame) # 흑백 전환

        board_img = extract_board(frame)
        board_state = classify_board(board_img, model)

        pred = None
        if board_state is not None:
            # (1, 9)을 (9,)로 변환 후 (3, 3)으로 reshape
            board_state_flat = board_state.flatten()  # (9,)로 변환
            board_state_reshaped = board_state_flat.reshape(3, 3)  # (9,) -> (3, 3)
            pred = board_state_reshaped.tolist()  # NumPy 배열을 리스트로 변환

        # 보드 상태 변화 감지
        if pred != prev_pred:
            change_detected = True
            last_change_time = time.time()
        else:
            change_detected = False

        # 안정화 조건 확인
        if not change_detected and time.time() - last_change_time > stability_duration:
            if pred != stable_pred:
                stable_pred = pred  # 안정화된 상태 저장
                print("Stable Board State Detected (O=0, X=1, Blank=2):")
                print(stable_pred)
        prev_pred = pred  # 현재 예측 값을 이전 예측 값에 저장

        cv2.imshow("Tic-Tac-Toe Board Detection", frame)

        # 종료 신호 확인
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting the program.")
            break
    
    vcap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
