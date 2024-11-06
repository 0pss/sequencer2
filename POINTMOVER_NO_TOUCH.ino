#include <FastLED.h>
#include <Wire.h>
#include <ClickEncoder.h>

// Original configuration remains unchanged
#define LED_PIN     5
#define NUM_LEDS    161
#define BRIGHTNESS  175
#define SATURATION  255

#define I2C_SLAVE_ADDRESS 0x08

#define ENCODER_PIN_A 4
#define ENCODER_PIN_B 5
#define ENCODER_BTN A3

#define ENCODER_STEPS_PER_BPM 4
#define ENCODER_COUNT 4
#define ENCODER_STEPS_PER_NOTCH 2

// Rest of the declarations remain the same
CRGB leds[NUM_LEDS];

struct Track {
    int start;
    int stop;
    int position;
    int direction;
};

Track tracks[4] = {
    {1, 40, 40, -2},
    {41, 80, 41, 2},
    {82, 120, 120, -2},
    {121, 160, 121, 2},
};

ClickEncoder encoder(ENCODER_PIN_A, ENCODER_PIN_B, ENCODER_BTN, ENCODER_STEPS_PER_NOTCH);
int16_t lastEncoderValue = 0;
int currentBPM = 120;

// Define a struct for task scheduling
typedef struct {
  uint32_t tick; // next execution time in milliseconds
  void (*task)(); // pointer to the task function
} Task;

// Global variables to store LED states
struct {
    byte animation_position;
    byte led_states[10];   // For storing direct LED control states from command 0x03
} led_control_state;

// Scheduling constants and buffer
#define TICKDEPTH 9
Task tickBuff[TICKDEPTH];
uint8_t tickHead = 0, tickTail = 0;

void addToTick(uint32_t newTick, void (*newTask)()) {
    uint8_t tHead = (tickHead + 1) % TICKDEPTH;
    if (tHead != tickTail) {
        tickBuff[tickHead].tick = newTick;
        tickBuff[tickHead].task = newTask;
        tickHead = tHead;
    }
}

void readEncoder() {
    int16_t encoderValue = encoder.getValue();
    if (encoderValue != 0) {
        int bpmChange = encoderValue / ENCODER_STEPS_PER_BPM;
        currentBPM = constrain(currentBPM + bpmChange, 40, 200);
        lastEncoderValue = encoderValue;
    }
}


void serviceEncoder(){
  encoder.service();
}


void update_leds() {
    FastLED.clear();
    
    // First layer: Set all direct-controlled LEDs (from command 0x03)
    for (int byte_index = 0; byte_index < 10; byte_index++) {
        byte led_states = led_control_state.led_states[byte_index];
        for (int bit_index = 0; bit_index < 8; bit_index++) {
            int led_index = byte_index * 8 + bit_index;
            if (led_index < NUM_LEDS) {
                bool led_on = (led_states & (1 << bit_index)) != 0;
                if (led_on) {
                    leds[led_index] = CRGB(255, 255, 0);
                }
            }
        }
    }
    
    // Second layer: Draw position animation on top (from command 0x01)
    for (int i = 0; i < 4; i++) {
        if (tracks[i].direction < 0) {
            tracks[i].position = tracks[i].stop + tracks[i].direction * led_control_state.animation_position;
        } else {
            tracks[i].position = tracks[i].start + tracks[i].direction * led_control_state.animation_position;
        }
        leds[tracks[i].position] = CRGB(255, 255, 0);
        
        if (tracks[i].position > tracks[i].stop) {
            tracks[i].position = tracks[i].start;
        } else if (tracks[i].position < tracks[i].start) {
            tracks[i].position = tracks[i].stop;
        }
    }
    
    FastLED.show();
}

void setup() {
    delay(1000);
    Wire.begin(I2C_SLAVE_ADDRESS);
    Wire.onReceive(receiveEvent);
    Wire.onRequest(requestEvent);
    
    FastLED.addLeds<WS2812B, LED_PIN, GBR>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);
    
    encoder.setAccelerationEnabled(true);

    // Schedule initial sensor and encoder tasks
    addToTick(millis() + 99, serviceEncoder);
    addToTick(millis() + 100, readEncoder);
    addToTick(millis() + 250, update_leds);

}

void loop() {
    // Process scheduled tasks
    if (tickTail != tickHead) {
        if (millis() > tickBuff[tickTail].tick) {
            tickBuff[tickTail].task(); // Execute the task
            tickTail = (tickTail + 1) % TICKDEPTH;
        }
    }

    // Re-schedule sensor and encoder reads
        addToTick(millis() + 99, serviceEncoder);
    addToTick(millis() + 100, readEncoder);
    addToTick(millis() + 250, update_leds);

    delay(10); // Small delay to avoid busy-waiting
}

void receiveEvent(int howMany) {
    if (howMany < 2) return;

    byte command = Wire.read();
    byte data = Wire.read();

    if (command == 0x01) {
        led_control_state.animation_position = data;
        update_leds();
        
    } 
}

void requestEvent() {

    Wire.write((byte)currentBPM);

}
