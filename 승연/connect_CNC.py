import serial
import time

# Initial Setup
# Set Serial port (포트를 정확히 확인하세요)
class Serial:
    def __init__(self, serial_port, is_first):
        self.ser = self.connect_serial(serial_port)
        self.OX = "O" if is_first else "X"

    def connect_serial(self, serial_port):
        '''
        Connect to the serial port and return it.
        '''
        ser = serial.Serial(serial_port, 115200, timeout=None)
        print("Communication Successfully started") # 연결 확인용
        return ser


    # 게임보드 그리는 명령을 보내는 함수
    def draw_game_board(self):
        '''
        Send 'S'(draw game board) command to CNC robot.
        '''
        while True:
            self.ser.write('S'.encode()) # 명령 전송
            print("Sent command: \'S\'")
            
            self.waiting_robot()
            break


    # 해당 신호를 보내는 함수
    def send_to_robot(self, first_command, second_command):
        '''
        Send 'first_command' and 'second_command' to the Arduino one after the other.
        first_command : 'O'/ 'X'
        second_command : action location (int 0~8)
        '''
        is_current_stage = True
        
        while True:
            time.sleep(1)
            if is_current_stage == True: # First command: O / X
                if first_command in ['O', 'X']:
                    self.ser.write(first_command.encode())  # 명령 전송
                    print(f"Sent command: {first_command}")

                    self.waiting_robot()

                    is_current_stage = False

                elif first_command == 'exit':
                    break

                else:
                    print("Invalid command")
                    break

                
            elif is_current_stage == False: # Second command: location
                if second_command in [str(i) for i in range(9)]:
                    self.ser.write(second_command.encode()) # Send command
                    print(f"Sent command: {second_command}")

                    self.waiting_robot()
                    
                    is_current_stage = True

                    break # 함수 종료

                elif second_command == 'exit':
                    break

                else:
                    print("Invalid command")
                    break

            time.sleep(1)

        return # 함수 종료


    def waiting_robot(self):
        '''
        Waiting while receiving a serial signal from the Arduino and reading it.
        '''
        while True:
            if self.ser.in_waiting > 0:  # 수신된 데이터가 있으면
                response = self.ser.readline().decode()  # 데이터 읽기
                print(f"Arduino Response: {response}")
                break

# # 코드 예시
# if __name__=="__main__":
#     ser = connect_serial('/dev/cu.usbserial-110') # 변수명 ser 변경 금지
#     waiting_robot()

#     draw_game_board()
#     # send_to_robot('X', '1')
#     # send_to_robot('O', '7')
    