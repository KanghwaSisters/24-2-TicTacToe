import cv2
import os

# 저장할 디렉토리 설정
save_dir = "captured_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 폴더 없으면 생성

# 사용 가능한 카메라 찾기
cap = None
for i in range(5):  # 0부터 4까지 시도
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"카메라 {i}가 열렸습니다.")
        break
    cap.release()

if not cap or not cap.isOpened():
    print("사용 가능한 카메라를 찾을 수 없습니다.")
    exit()

# 캡처 이미지 카운트
capture_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 영상 출력
    cv2.imshow('Webcam', frame)

    key = cv2.waitKeyEx(1)  # Mac에서 키 입력 감지를 개선

    if key == ord('q'):  # 'q' 키를 누르면 종료
        print("프로그램 종료")
        break
    elif key == ord('c'):  # 'c' 키를 누르면 이미지 캡처
        capture_count += 1
        img_name = os.path.join(save_dir, f"capture_{capture_count}.png")
        cv2.imwrite(img_name, frame)
        print(f"이미지 저장 완료: {img_name}")

# 자원 해제
cap.release()
cv2.destroyAllWindows()