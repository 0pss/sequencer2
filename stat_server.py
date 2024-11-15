from flask import Flask, jsonify, session, render_template
import random
import socket
import os
import psutil
from managers import SequencerState

from flask import g


app = Flask(__name__)

app.g_state = 0

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def data():
    state = app.g_state
    if state is not 0:
        bpm = state.bpm.value
    else:
        bpm = 666  # Default value if no state is found

    cpu_load = psutil.cpu_percent(percpu=True)
    graph_data = {
        "x": list(range(len(cpu_load))),
        "y": cpu_load
    }
    
    return jsonify({
        "numbers": cpu_load,
        "graph_data": graph_data,
        "bpm": bpm
    })

def get_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip

def serve(state: SequencerState):
    local_ip = get_local_ip()
    port = 5000
    app.g_state = state  # Store the state in session
    print(f"Visit the app on your phone at: http://{local_ip}:{port}")
    app.run(host='0.0.0.0', port=port)



def serve_with_realtime_priority(state: SequencerState):
    process = psutil.Process(os.getpid())
    
    # Set CPU affinity to core 1
    process.cpu_affinity([2])
    
    # Optionally set high priority (Unix/Linux only)
    try:
        os.nice(-19)  # Highest priority
    except Exception as e:
        print(f"Could not set process priority: {e}")
        
    serve(state)