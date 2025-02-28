import serial
import time

# 시리얼 포트 설정 (포트를 정확히 확인하세요)
ser = serial.Serial('/dev/cu.usbserial-110', 115200)
print("Communication Successfully started")

current_state = 0

while True:
    if current_state == 0: # First command: draw game board / O / X
        first_command = input("'S'(draw game board) / 'O' / 'X' :")

        if first_command in ['S', 'O', 'X']:
            ser.write(first_command.encode())  # 명령 전송
            print(f"Sent command: {first_command}")

            # Arduino로부터 완료 신호 대기
            while True:
                if ser.in_waiting > 0:  # 수신된 데이터가 있으면
                    response = ser.readline().decode().strip()  # 데이터 읽기
                    print(f"Arduino Response: {response}")
                    break

            current_state = 0 if first_command == 'S' else 1

        elif first_command == 'exit':
            break

        else:
            print("Invalid command")


    elif current_state == 1: # Second command: location
        second_command = input("Enter the location(0 ~ 8) :")

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

        elif second_command == 'exit':
            break

        else:
            print("Invalid command")

    time.sleep(1)
