#include <FastLED.h>
#include <Wire.h>
#include <Encoder.h>

// Original configuration remains unchanged
#define LED_PIN     5
#define NUM_LEDS    161
#define BRIGHTNESS  175
#define SATURATION  255

#define I2C_SLAVE_ADDRESS 0x08

Encoder myEnc(2, 3);
long oldPosition  = -999;
long newPosition = 0;

bool sequencer_on[4][16];

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

void serviceEncoder(){
    newPosition = myEnc.read()/4;
}


void update_leds() {
    FastLED.clear();
    
    // First layer: Set LEDs based on sequencer states
    for (int row = 0; row < 4; row++) {
        Track& track = tracks[row];
        
        // Calculate the number of usable positions in this track
        int track_length = (abs(track.stop - track.start) / 2) + 1;
        
        // Map each sequencer column to a physical LED position
        for (int col = 0; col < 16; col++) {
            if (sequencer_on[row][col]) {
                // Calculate physical LED position based on track direction and start
                int led_index;
                if (track.direction > 0) {
                    // For positive direction tracks, start from left
                    led_index = track.start + (col * 2);  // Skip every other LED
                } else {
                    // For negative direction tracks, start from right
                    led_index = track.stop - (col * 2);   // Skip every other LED
                }
                
                // Only set LED if it's within the track's bounds
                if (led_index >= track.start && led_index <= track.stop) {
                    leds[led_index] = CRGB(0, 80, 255);  // Red-orange for sequencer
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
        leds[tracks[i].position] = CRGB(255, 255, 0);  // Yellow color for animation
        
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

    Serial.begin(115200);
    
    FastLED.addLeds<WS2812B, LED_PIN, GBR>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);
    

    // Schedule initial sensor and encoder tasks
    addToTick(millis() + 49, serviceEncoder);
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
    addToTick(millis() + 49, serviceEncoder);
    addToTick(millis() + 250, update_leds);

    delay(10); // Small delay to avoid busy-waiting
}

void receiveEvent(int howMany) {
    Serial.print("Received bytes: ");
    Serial.println(howMany);
    
    // Print all received bytes in hex format
    Serial.print("Raw data: ");
    for(int i = 0; i < howMany; i++) {
        byte data = Wire.peek();
        Serial.print("0x");
        if(data < 16) Serial.print("0");  // Add leading zero for single digit hex
        Serial.print(data, HEX);
        Serial.print(" ");
    }
    Serial.println();
        
    if (howMany < 1) {
        Serial.println("Error: No data received");
        return;
    }
    
    byte command = Wire.peek();  // Look at first byte without removing it
    Serial.print("Command byte: 0x");
    Serial.println(command, HEX);
    
    // Process based on first byte
    if (command == 0x01) {
        Serial.println("Animation position command received");
        Wire.read();  // Remove command byte
        if (Wire.available()) {
            led_control_state.animation_position = Wire.read();
            Serial.print("New animation position: ");
            Serial.println(led_control_state.animation_position);
            update_leds();
        }
    }
    else if (command == 0x02) {
        Serial.println("Sequencer state command received");
        Wire.read();  // Remove command byte
        
        if (Wire.available() >= 8) {  // Check if we have enough data
            Serial.println("Reading sequencer states...");
            for (int row = 0; row < 4; row++) {
                byte byte1 = Wire.read();
                byte byte2 = Wire.read();
                
                Serial.print("Row ");
                Serial.print(row);
                Serial.print(" bytes: 0x");
                Serial.print(byte1, HEX);
                Serial.print(" 0x");
                Serial.println(byte2, HEX);
                
                // Fill the row with the bits from the two bytes
                for (int col = 0; col < 8; col++) {
                    sequencer_on[row][col] = (byte1 >> col) & 1;
                    sequencer_on[row][col + 8] = (byte2 >> col) & 1;
                }
            }
            
            // Print the resulting grid
            Serial.println("Resulting sequencer grid:");
            for (int row = 0; row < 4; row++) {
                Serial.print("Row ");
                Serial.print(row);
                Serial.print(": ");
                for (int col = 0; col < 16; col++) {
                    Serial.print(sequencer_on[row][col]);
                    Serial.print(" ");
                }
                Serial.println();
            }
        } else {
            Serial.print("Error: Expected 8 bytes of data, but only ");
            Serial.print(Wire.available());
            Serial.println(" bytes available");
        }
    }
    else {
        Serial.print("Unknown command received: 0x");
        Serial.println(command, HEX);
        // Flush remaining bytes
        while(Wire.available()) {
            Wire.read();
        }
    }
    
}

void requestEvent() {

    Serial.println("BPM SENT");

    long bpmValue = newPosition; // Example long value

    // Split the long into individual bytes
    byte byte1 = (byte)(bpmValue & 0xFF);
    byte byte2 = (byte)((bpmValue >> 8) & 0xFF);
 

    // Send the bytes one by one
    Wire.write(byte1);
    Wire.write(byte2);

}
