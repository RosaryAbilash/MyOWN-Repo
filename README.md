# MyOWN-Repo
CODES


from time import sleep
from max30102 import MAX30102

# Initialize the MAX30102 sensor
max30102 = MAX30102()

try:
    # Read pulse input continuously
    while True:
        # Read the red-led and IR-led data sequentially
        red_data, ir_data = max30102.read_sequential()

        # Process the data as needed
        # For example, calculate heart rate or SpO2 level
        
        # Print the data for demonstration
        print("Red data: {}".format(red_data))
        print("IR data: {}".format(ir_data))

        # Add a short delay before the next reading
        sleep(1)

except KeyboardInterrupt:
    # Handle the case where the user interrupts the program
    print("Program terminated by user.")
finally:
    # Shutdown the MAX30102 sensor
    max30102.shutdown()
