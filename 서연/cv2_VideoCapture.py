import cv2

def initialize_camera(camera_id=0):

    # 1. 카메라 열기
    # 0 - 시스템 기본 카메라
    vcap = cv2.VideoCapture(camera_id) # 클래스 생성

    # 2. 카메라가 열렸는지 확인
    if not vcap.isOpened():
        raise IOError("Camera open failed!")
    
    return vcap

def get_frame(vcap):
    # 3. 프레임 받아오기 - read()
    # read() -> retval, image 반환
    # ret: 성공하면 True, 실패하면 False
    # image: 현재 프레임 (numpy.ndarray)

    ret, frame = vcap.read()

    if not ret: # 새로운 프레임 X
        return None

    return frame
