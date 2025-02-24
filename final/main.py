import time
import numpy as np
import torch
from square_board_recognition import camera_to_state

def main():
    """
    camer_to_state 함수 test
    """

    # 초기 state 설정 (예시: 빈 보드 상태)
    initial_state = np.array([[0, 2, 2, 2, 1, 1, 2, 2, 0]])  # 2는 빈 칸을 나타냄

    # detect_and_stabilize_board_state 함수 호출하여 안정화된 보드 상태 반환
    stable_state = camera_to_state(initial_state)

    if stable_state is not None:
        print("Stable Board State Detected:")
        print(stable_state)
    else:
        print("No stable state detected.")

if __name__ == "__main__":
    main()