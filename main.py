# -*- coding: utf-8 -*-

from cv2_VideoCapture import initialize_camera, get_frame
from board_recognition import TicTacToeCNN, extract_board, classify_board, classify_cell
import cv2
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TicTacToeCNN().to(device)
    model.load_state_dict(torch.load(r"C:\Temp\model_state_dict.pt", map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    vcap = initialize_camera(0)

    while True:
        frame = get_frame(vcap)
        if frame is None:
            break

        # frame = cv2.bitwise_not(frame)

        board_img = extract_board(frame)
        board_state = classify_board(board_img, model)

        if board_state is not None:
            print("Predicted board state:")
            for row in board_state:
                print(" | ".join(["O" if cell == 0 else "X" if cell == 1 else " " for cell in row]))

        cv2.imshow("Tic-Tac-Toe Board Detection", frame)

        if cv2.waitKey(10) == 27:
            break

    vcap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()