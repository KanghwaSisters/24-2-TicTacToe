# import
import pickle
import deque

from ResNet import Net

# history 불러오는 함수
MEM_SIZE = 50000

def load_history(file):
    '''
    file 경로에 저장되어 있는 history를 불러오는 함수
    '''
    try:
        with open(file, 'rb') as f:
            history = pickle.load(f)
    except FileNotFoundError:
        history = deque(maxlen = MEM_SIZE) # 파일이 비어있는 경우 빈 리스트 생성

    return history

# history 저장 함수
def save_history(file, data):
    '''
    file 경로에 history를 저장하는 함수
    '''
    with open(file, 'wb') as f:
        pickle.dump(data, f)

# model 파라미터를 가져오는 함수
def load_model(file):
    '''
    file 경로에 저장되어 있는 모델 파라미터로 모델을 불러오는 함수
    '''
    model = Net()
    try:
        with open(file, 'rb') as f:
            model.load_state_dict(pickle.load(f))
    except FileNotFoundError:
        pass
    return model

# 최근 모델 저장 함수
def save_model(file, model):
    '''
    file 경로에 모델 파라미터를 저장하는 함수
    '''
    with open(file, 'wb') as f:
        pickle.dump(model.state_dict(), f)

