import serial
import time

# Initial Setup
# Set Serial port (포트를 정확히 확인하세요)
ser = serial.Serial('/dev/cu.usbserial-110', 115200, timeout=None)
print("Communication Successfully started") # 연결 확인용

def draw_game_board():
    '''
    Send 'S'(draw game board) command to CNC robot.
    '''
    while True:
        ser.write('S'.encode()) # 명령 전송
        print("Sent command: \'S\'")
        waiting_robot()
        break


# 해당 신호를 보내는 함수
def send_to_robot(first_command, second_command):
    '''
    first_command : 'O'/ 'X'
    second_command : action location (int 0~8)
    '''
    current_state = 0
    
    while True:
        if current_state == 0: # First command: O / X
            if first_command in ['O', 'X']:
                ser.write(first_command.encode())  # 명령 전송
                print(f"Sent command: {first_command}")

                waiting_robot()

                current_state = 1

            elif first_command == 'exit':
                break

            else:
                print("Invalid command")
                break


        elif current_state == 1: # Second command: location
            if second_command in [str(i) for i in range(9)]:
                ser.write(second_command.encode()) # Send command
                print(f"Sent command: {second_command}")

                waiting_robot()
                
                current_state = 0

                break # 함수 종료

            elif second_command == 'exit':
                break

            else:
                print("Invalid command")
                break

        time.sleep(1)

    return # 함수 종료


def waiting_robot():
    while True:
        if ser.in_waiting > 0:  # 수신된 데이터가 있으면
            response = ser.readline().decode()  # 데이터 읽기
            print(f"Arduino Response: {response}")
            break

# 코드 예시
if __name__=="__main__":
    ser = serial.Serial('/dev/cu.usbserial-110', 115200, timeout=None)
    print("Communication Successfully started") # 연결 확인용
    waiting_robot()
        
    draw_game_board()
    send_to_robot('X', '1')
    send_to_robot('O', '7')
    