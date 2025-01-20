# -*- coding: utf-8 -*-

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TicTacToeCNN(nn.Module):
    def __init__(self):
        super(TicTacToeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256*4*4, 128)
        self.fc2 = nn.Linear(128, 9*3)  # O, X, blank (3 classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        x = x.view(x.size(0), 9, 3)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
])

def preprocess_image(frame):
    input_tensor = transform(frame).unsqueeze(0).to(device)
    return input_tensor

def classify_cell(model, frame):
    input_tensor = preprocess_image(frame)
    with torch.no_grad():
        output = model(input_tensor)
    board_state = torch.argmax(output, dim=2).cpu().numpy()
    return board_state

def extract_square_board(frame):
    """
    카메라 각도 때문에 비정형 사각형으로 인식되는 틱택토 보드를 
    정사각형으로 보정하여 인식
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 특정 임계값 이하의 픽셀을 흰색으로 설정
    shadow_threshold = 100 # 그림자를 제거하기 위한 임계값
    gray[gray < shadow_threshold] = 255 # 임계값 이하를 흰색으로 설정
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    """
    equalized_gray = cv2.equalizeHist(gray)

    blurred = cv2.GaussianBlur(equalized_gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    """


    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 5000:
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) == 4:
                points = np.array([point[0] for point in approx])
                sorted_points = order_points(points)

                # 디버킹 (찾은 꼭짓점 시각화)
                for corner in sorted_points:
                    x, y = corner
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)

                square_size = 300
                dst_points = np.array([
                    [0, 0],
                    [square_size - 1, 0],
                    [square_size - 1, square_size - 1],
                    [0, square_size - 1]
                    ], dtype="float32")

                # 투시 변환 행렬 계산
                matrix = cv2.getPerspectiveTransform(sorted_points, dst_points)
                board_img = cv2.warpPerspective(frame, matrix, (square_size, square_size))

            _, board_img_thresh = cv2.threshold(cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY), 
                                                128, 255, cv2.THRESH_BINARY)
            board_img_colored = cv2.cvtColor(board_img_thresh, cv2.COLOR_GRAY2BGR)

            # 디버깅: 보정된 보드와 원본 프레임 표시
            """
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Extracted Square Board", board_img_colored)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)

            return board_img_colored
    return None

def order_points(points):
    """ 
    좌표 정렬: 좌상, 우상, 우하, 좌하 순서
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)] # 좌상
    rect[2] = points[np.argmax(s)] # 우하

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)] # 우상
    rect[3] = points[np.argmax(diff)] # 좌하

    return rect

def visualize_corners(frame, corners):
    """
    4개의 꼭짓점을 시각화하여 보여주는 함수
    """
    # 꼭짓점에 원을 그림
    for corner in corners:
        x, y = corner[0]
        cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)  # 빨간색 원을 그리기

    # 꼭짓점들을 선으로 연결
    for i in range(4):
        pt1 = tuple(corners[i][0])
        pt2 = tuple(corners[(i+1) % 4][0])
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # 초록색 선으로 연결

    return frame

def classify_board(board_img, model):
    board_img_resized = cv2.resize(board_img, (96, 96))
    board_state = classify_cell(model, board_img_resized)
    return board_state

"""
def extract_cells(frame, board_contour):
    x, y, w, h = cv2.boundingRect(board_contour)
    board = frame[y:y+h, x:x+w]
    cell_size = h // 3
    cells_with_contours = []

    for i in range(3):
        for j in range(3):
            cell = board[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_cell, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(cell, contours, -1, (0, 0, 255), 2)

            cells_with_contours.append((i, j, cell))

    return cells_with_contours
"""
