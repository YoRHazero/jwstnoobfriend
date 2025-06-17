from jwstnoobfriend.utils.display import track_func, time_footer
import time
from rich.layout import Layout
from rich.console import Console
from rich.live import Live
from rich.panel import Panel


@track_func(progress_paramkey='items', refresh_per_second=5, progress_description='Processing items...')
def check_track_func(items: list):
    result = []
    for item in items:
        time.sleep(0.01)
        result.append(item)
    return result

def check_time_footer():
    console = Console()
    layout = Layout()
    layout.split_column(
        Layout(name="main",),
        Layout(name="footer", size=3)
    )
    layout["main"].update(Panel("Main content goes here", title="Main Panel"))
    @time_footer(layout["footer"])
    def do_some_work():
        with Live(layout, refresh_per_second=10) as live:
            for i in range(100):
                layout["main"].update(Panel(f"Processing step {i+1}/100"))
                time.sleep(0.1)

    do_some_work()
            
    
    
    


if __name__ == "__main__":
    items = [i for i in range(100)]
    result = check_track_func(items)
    check_time_footer()
    
