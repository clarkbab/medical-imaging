import debugpy
import json
import os
import socket
from typing import *

def debug(port: int = 5679) -> None:
    # This will be called from within a notebook.

    # Create 'launch.json'.
    host = socket.gethostname()
    config = launch_template(host, port)
    filepath = os.path.join(os.environ['MYMI_CODE'], '.vscode', 'launch.json') 
    print(filepath)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

    # Start debug server.
    debugpy.listen(("0.0.0.0", port))
    print('Waiting for debug client..')
    debugpy.wait_for_client()
    print('Client attached.')

def launch_template(
    host: str,
    port: int) -> Dict:
    return {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Debug: Jupyter Notebook",
                "type": "debugpy",
                "request": "attach",
                "host": host,          
                "port": port,
                "justMyCode": True,
            }
        ]
    }
