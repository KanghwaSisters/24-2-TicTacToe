import cv2

def initialize_camera():
    """
    ������ camera_id�� ī�޶� ���� cv2.VideoCapture ��ü�� ��ȯ�ϴ� �Լ�.
    ī�޶� ���������� ������ ������ IOError ���� �߻�.
    """
    # 1. ī�޶� ����
    # 0 - �ý��� �⺻ ī�޶�
    vcap = None
    for i in range(5):  # 0부터 4까지 시도
        vcap = cv2.VideoCapture(i)
        if vcap.isOpened():
            print(f"카메라 {i}가 열렸습니다.")
            break
        # vcap.release()

    return vcap

def get_frame(vcap):
    """
    vcap ��ü���� �� �������� ĸó�Ͽ� ��ȯ�ϴ� �Լ�.
    �������� ���������� �������� ���ϸ� None�� ��ȯ.
    """
    # 3. ������ �޾ƿ��� - read()
    # read() -> retval, image ��ȯ
    # ret: �����ϸ� True, �����ϸ� False
    # image: ���� ������ (numpy.ndarray)

    ret, frame = vcap.read()

    if not ret: # ���ο� ������ X
        return None

    return frame
