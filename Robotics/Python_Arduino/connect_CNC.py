import serial
import time

# Initial Setup
# Set Serial port (포트를 정확히 확인하세요)
ser = serial.Serial('/dev/cu.usbserial-110', 115200)
print("Communication Successfully started") # 연결 확인용

# 해당 신호를 보내는 함수
def send_to_robot(first_command, second_command):
    '''
    first_command : 'start'(draw game board) / 'O'/ 'X'
    second_command : action location (int 0~8)
    Note: If you command "draw game board", input 'exit' in second_command.
    '''
    current_state = 0
    
    while True:
        if current_state == 0: # First command: draw game board / O / X
            if first_command in ['start', 'O', 'X']:
                ser.write(first_command.encode())  # 명령 전송
                print(f"Sent command: {first_command}")

                # Arduino로부터 완료 신호 대기
                while True:
                    if ser.in_waiting > 0:  # 수신된 데이터가 있으면
                        response = ser.readline().decode().strip()  # 데이터 읽기
                        print(f"Arduino Response: {response}")
                        break

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

                # Arduino로부터 완료 신호 대기
                while True:
                    if ser.in_waiting > 0:  # 수신된 데이터가 있으면
                        response = ser.readline().decode().strip()  # 데이터 읽기
                        print(f"Arduino Response: {response}")
                        break
                
                current_state = 0

                break # 함수 종료

            elif second_command == 'exit':
                break

            else:
                print("Invalid command")
                break

        time.sleep(1)

    return # 함수 종료
