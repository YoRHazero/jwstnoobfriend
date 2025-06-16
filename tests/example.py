from rich.layout import Layout
from rich.console import Console
from rich.live import Live
from rich.table import Table
import time
from datetime import timedelta
import threading

layout = Layout()
layout.split(
    Layout(name='main'),
    Layout(name='footer', size =2)
)
start_time = time.time()
stop_timer = threading.Event()
def update_footer_time():
    while not stop_timer.is_set():
        elapsed_time = time.time() - start_time
        layout["footer"].update