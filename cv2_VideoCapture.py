import cv2

def initialize_camera(camera_id=0):

    # 1. ī�޶� ����
    # 0 - �ý��� �⺻ ī�޶�
    vcap = cv2.VideoCapture(camera_id) # Ŭ���� ����

    # 2. ī�޶� ���ȴ��� Ȯ��
    if not vcap.isOpened():
        raise IOError("Camera open failed!")
    
    return vcap

def get_frame(vcap):
    # 3. ������ �޾ƿ��� - read()
    # read() -> retval, image ��ȯ
    # ret: �����ϸ� True, �����ϸ� False
    # image: ���� ������ (numpy.ndarray)

    ret, frame = vcap.read()

    if not ret: # ���ο� ������ X
        return None

    return frame
