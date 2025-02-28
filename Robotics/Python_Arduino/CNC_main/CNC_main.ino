#include "StepMotor.h"

Stepper stepper_x(8, 9, 10, 11);
Stepper stepper_y(3, 4, 5, 6);
Stepper stepper_z(A5, A4, A3, A2);

TicTacToeArtist artist(21, 21, &stepper_x, &stepper_y, &stepper_z);

// int current_state = 0; // 현재 상태를 추적하는 변수

char type;          // 첫 번째 문자를 저장할 변수
int position;       // 두 번째 숫자를 저장할 변수

void setup() {
    Serial.begin(9600); // 시리얼 통신 시작
    Serial.println("Setup complete");
}

void loop() {
    if (Serial.available()) { // 새로운 데이터가 들어왔는지 확인
        String command = Serial.readStringUntil('\n');  // 한 줄 읽기 (줄바꿈 문자까지)

        type = command.charAt(0);  // 첫 번째 문자 추출
        position = command.substring(1).toInt();  // 두 번째 이후 숫자를 정수로 변환

        switch (type) {
        case 'X':
            artist.drawX(position);
            Serial.println("Done: drawX");
            break;
        
        case 'O':
            artist.drawCircle(position);
            Serial.println("Done: drawCircle");
            break;
          
        case 'S':
            artist.drawGameBoard();
            Serial.println("Done: drawGameBoard");
        }


        // switch (current_state) {
        //     case 0 :
        //         switch (command) {
        //             case 'S' :
        //                 artist.drawGameBoard();
        //                 Serial.println("Done: draw Game board");
        //                 current_state = 0;
        //                 break;

        //             case 'O' :
        //                 current_state = 1;
        //                 Serial.println("Setting O");
        //                 break;

        //             case 'X' :
        //                 current_state = 2;
        //                 Serial.println("Setting X");
        //                 break;

        //             default:
        //                 Serial.println("Nothing");
        //         }
        //         break;

        //     case 1 :
        //         int_command = command;
        //         artist.drawCircle(int_command);
        //         Serial.println("Done: drawCircle");
        //         current_state = 0;
        //         break;
            
        //     case 2 :
        //         int_command = command;
        //         artist.drawX(int_command);
        //         Serial.println("Done: drawX");
        //         current_state = 0;
        //         break;

        //     default:
        //         Serial.println("Nothing");
        // }
    }
}
