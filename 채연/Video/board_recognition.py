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
        self.dropout = nn.Dropout(0.5)
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

def extract_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE로 명암비 개선
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced_gray, 
        100, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )

    # Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)
    cleaned = cv2.erode(dilated, kernel, iterations=1)

    # Canny Edge Detection
    edges = cv2.Canny(cleaned, 50, 150)

    # Contour Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 필터링: 가장 큰 컨투어 추출
    filtered_contours = [c for c in contours if 10000 < cv2.contourArea(c) < 100000]
    if filtered_contours:
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        if 0.8 <= w / h <= 1.2:  # 정사각형 비율 확인
            board_img = frame[y:y+h, x:x+w]
            return board_img
    return None


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