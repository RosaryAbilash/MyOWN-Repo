# import streamlit as st
# import time

# # Function to simulate connecting to WiFi interface
# def connect_to_interface():
#     st.write("Connecting to WiFi Interface...", unsafe_allow_html=True)
#     time.sleep(2)  # Simulating connection time
#     st.write("<span style='color: green;'>Connected to WiFi Interface</span>", unsafe_allow_html=True)


# # Function to simulate receiving packets
# def receive_packets():
#     st.write("Receiving Packets from the Interface...", unsafe_allow_html=True)
#     time.sleep(10)  # Simulating packet receiving time
#     flag = 1
#     st.write("<span style='color: green;font-size: 20px;'>Packets Received</span>", unsafe_allow_html=True)


# # Function to read and display captured packets from the text file
# def show_captured_packets():
#     with open('captured_packets.txt', 'r') as file:
#         captured_packets = file.read()
#     st.write("<span style='color: blue;'>Captured Packets:</span>", unsafe_allow_html=True)
#     st.write(captured_packets, unsafe_allow_html=True)


# # Function to simulate detecting malware
# def detect_malware():
#     if flag == 1:
#         st.write("Detecting Malware...", unsafe_allow_html=True)
#         time.sleep(20)  # Simulating detection time
#         st.write("<span style='color: red; font-size: 20px;'>No malware detected</span>", unsafe_allow_html=True)
#     else:
#         st.write("Recieve Packets Before Malware Detection...", unsafe_allow_html=True)

# # Streamlit app layout
# def main():
#     flag = 0
#     st.title("IoT Malware Detection")
    
#     # Button 1: Connect to WiFi Interface
#     st.write("<div style='display:flex;justify-content:center;'>", unsafe_allow_html=True)
#     if st.button("Connect to WiFi Interface"):
#         connect_to_interface()
        
#     # Button 2: Receive Packets
#     if st.button("Receive Packets"):
#         receive_packets()
            
#     # Button 3: Show Captured Packets
#     if st.button("Show Captured Packets"):
#         show_captured_packets()
            
#     # Button 4: Detect Malware
#     if st.button("Detect Malware"):
#         detect_malware()

#     st.write("</div>", unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()





import streamlit as st
import time

# Function to simulate connecting to WiFi interface
def connect_to_interface():
    st.write("Connecting to WiFi Interface...", unsafe_allow_html=True)
    time.sleep(2)  # Simulating connection time
    st.session_state.flag_1 = 1
    st.write("<span style='color: green;'>Connected to WiFi Interface</span>", unsafe_allow_html=True)


# Function to simulate receiving packets
def receive_packets():
    if st.session_state.flag_1:
        st.write("Receiving Packets from the Interface...", unsafe_allow_html=True)
        time.sleep(10)  # Simulating packet receiving time
        st.session_state.flag = 1  # Set flag to 1 indicating packets received
        st.write("<span style='color: green;font-size: 20px;'>Packets Received</span>", unsafe_allow_html=True)
    else:
        st.write("<span style='color: red;font-size: 20px;'>Connect Interface Before Recieve Packets...</span>", unsafe_allow_html=True)



# Function to read and display captured packets from the text file
def show_captured_packets():
    with open('captured_packets.txt', 'r') as file:
        captured_packets = file.read()
    st.write("<span style='color: blue;'>Captured Packets:</span>", unsafe_allow_html=True)
    st.write(captured_packets, unsafe_allow_html=True)


# Function to simulate detecting malware
def detect_malware():
    if st.session_state.flag == 1:  # Check if packets have been received
        st.write("Detecting Malware...", unsafe_allow_html=True)
        time.sleep(20)  # Simulating detection time
        st.write("<span style='color: red; font-size: 20px;'>No malware detected</span>", unsafe_allow_html=True)
    else:
        st.write("Receive Packets Before Malware Detection...", unsafe_allow_html=True)

# Streamlit app layout
def main():
    st.title("IoT Malware Detection")
    
    # Initialize session state
    if 'flag' not in st.session_state:
        st.session_state.flag = 0

    if 'flag_1' not in st.session_state:
        st.session_state.flag_1 = 0
    
    # Button 1: Connect to WiFi Interface
    st.write("<div style='display:flex;justify-content:center;'>", unsafe_allow_html=True)
    if st.button("Connect to WiFi Interface"):
        connect_to_interface()
        
    # Button 2: Receive Packets
    if st.button("Receive Packets"):
        receive_packets()
            
    # Button 3: Show Captured Packets
    if st.button("Show Captured Packets"):
        show_captured_packets()
            
    # Button 4: Detect Malware
    if st.button("Detect Malware"):
        detect_malware()

    st.write("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
