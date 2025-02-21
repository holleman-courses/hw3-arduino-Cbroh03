#include <Arduino.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "sine_model_int8.h"

#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 7
#define TENSOR_ARENA_SIZE 2 * 1024

// Declare TFLM components
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];
const tflite::Model *model;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input;
TfLiteTensor *output;

// Function declarations
int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);
int sum_array(int *int_array, int array_len);
void run_inference(int *input_array);
unsigned long measure_time();

char received_char = (char)NULL;
char out_str_buff[OUTPUT_BUFFER_SIZE];
char in_str_buff[INPUT_BUFFER_SIZE];
int input_array[INT_ARRAY_SIZE];
int in_buff_idx = 0;

void setup() {
  Serial.begin(115200);
  delay(5000);
  Serial.println("Test Project waking up");

  // Load the TFLite model
  model = tflite::GetModel(sine_model_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &micro_error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }
  
  input = interpreter->input(0);
  output = interpreter->output(0);
  memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
}

void loop() {
  if (Serial.available() > 0) {
    received_char = Serial.read();
    Serial.print(received_char);
    in_str_buff[in_buff_idx++] = received_char;
    
    if (received_char == 13) { // Enter key pressed
      Serial.println("Processing input...");
      
      int array_length = string_to_array(in_str_buff, input_array);
      if (array_length != INT_ARRAY_SIZE) {
        Serial.println("Error: Please enter exactly 7 numbers.");
      } else {
        print_int_array(input_array, array_length);
        run_inference(input_array);
      }
      
      memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
      in_buff_idx = 0;
    } else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
      memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
      in_buff_idx = 0;
    }
  }
}

int string_to_array(char *in_str, int *int_array) {
  int num_integers = 0;
  char *token = strtok(in_str, ",");
  while (token != NULL) {
    int_array[num_integers++] = atoi(token);
    token = strtok(NULL, ",");
    if (num_integers >= INT_ARRAY_SIZE) break;
  }
  return num_integers;
}

void print_int_array(int *int_array, int array_len) {
  Serial.print("Integers: [");
  for (int i = 0; i < array_len; i++) {
    Serial.print(int_array[i]);
    if (i < array_len - 1) Serial.print(", ");
  }
  Serial.println("]");
}

void run_inference(int *input_array) {
  unsigned long t0, t1, t2;

  // Measure print time
  t0 = micros();
  Serial.println("Running inference...");
  Serial.println("test statement");
  t1 = micros();
  
  // Load input into model
  for (int i = 0; i < INT_ARRAY_SIZE; i++) {
    input->data.int8[i] = (int8_t)((input_array[i] - 0.0f) * (255.0f / 6.0f) - 128.0f);  // Scale to [-128, 127]
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }

  t2 = micros();
  
  // Get the model's prediction
  int8_t result = output->data.int8[0];
  result = result / 32.0f;
  Serial.print("Prediction: ");
  Serial.println(result);
  
  // Print timing results
  Serial.print("Printing time = ");
  Serial.print(t1 - t0);
  Serial.print(" us. Inference time = ");
  Serial.print(t2 - t1);
  Serial.println(" us.");
}
