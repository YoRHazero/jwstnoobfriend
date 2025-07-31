import inspect
import functools
from rich.progress import Progress, TaskID, track
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from pydantic import validate_call
from typing import Callable, Iterable, Literal
import threading
import time
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


__all__ = ['console','track_func','time_footer', 'plotly_figure_and_mask']
## Terminal part
console = Console()
    
@validate_call
def track_func(
    progress_paramkey: str | None = None, 
    refresh_per_second: int = 10, 
    progress_description: str = "Processing ...",
) -> Callable:
    """
    A decorator to add a progress bar to a function that processes an iterable parameter.
    
    Parameters
    -----------
    progress_paramkey: str | None
        The key of the parameter in the function that is an iterable to track progress on.
        If None, it will automatically find the first iterable parameter.
    
    refresh_per_second: int
        The number of times the progress bar will refresh per second.
    
    progress_description: str
        The description to display in the progress bar.
        
    Returns
    --------
    Callable
        A decorator that wraps the function to add a progress bar.
    
    Example
    --------

    """    
    def progress_bar_decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal progress_paramkey
            sig = inspect.signature(func)
            func_params = sig.bind(*args, **kwargs)
            func_params.apply_defaults()
            
            # Check if the progress_paramkey is provided, if not, set it to the first iterable parameter
            if progress_paramkey is None:
                progress_paramkey = [p_key 
                                      for p_key, p_val in func_params.arguments.items()
                                      if isinstance(p_val, Iterable) and not isinstance(p_val, str)][0]
            # Validate that the progress_paramkey exists in the function arguments
            if progress_paramkey not in func_params.arguments:
                raise ValueError(f"Progress parameter '{progress_paramkey}' not found in function arguments.")
            
            progress_interable = func_params.arguments[progress_paramkey]
            with Progress(refresh_per_second=refresh_per_second) as progress:
                task = progress.add_task(progress_description, total=len(progress_interable))
                
                def inner_wrapper(progress_iterable: Iterable):
                    for item in progress_iterable:
                        yield item
                        progress.update(task, advance=1)
                
                # Replace the iterable parameter with a generator that updates the progress bar
                func_params.arguments[progress_paramkey] = inner_wrapper(progress_interable)
                result = func(*func_params.args, **func_params.kwargs)
                return result
        return wrapper
    return progress_bar_decorator


## to do list:
# 1. include the logic of splitting the layout and updating with live in the decorator

def time_footer(func: Callable) -> Callable:
    @functools.wraps(func)  # wraps is also not a decorator, but a decorator factory
    def wrapper(*args, **kwargs):
        start_time = time.time()
        with Live(
            refresh_per_second=2,
        ) as live:
            
            def update_time(refresh_per_second: int = 2):
                while True:
                    elapsed_time = time.time() - start_time
                    formatted_time = str(timedelta(seconds=int(elapsed_time)))
                    timer_text = Text(f"⏱️ Running Time: {formatted_time}")
                    live.update(Panel(timer_text))
                    time.sleep(1 / refresh_per_second)
            
            threading.Thread(target=update_time, daemon=True).start()
            
            original_print = console.print
            def live_print(*renderables, **kwargs):
                live.console.print(*renderables, **kwargs)
            
            console.print = live_print
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore the original print function
                console.print = original_print
    return wrapper
            

# Visualization part

def plotly_figure_and_mask(
    data: list[np.ndarray] | None = None,
    mask: list[np.ndarray] | None = None,
    pmin: float = 1.0,
    pmax: float = 99.0,
    zmin: float | None = None,
    zmax: float | None = None,
    cmap: str = 'gray',
    binary_mode: bool = True,
    height: int = 600,
    width: int = 600,
    align_mode: Literal['blink', 'wrap'] = 'animate',
    subtitles: list[str] | None = None,
) -> go.Figure:
    """
    Create a Plotly figure with a list of data and mask arrays, applying a color scale and alignment.
    
    Parameters
    ----------
    data : list[np.ndarray] | None
        A list of numpy arrays representing the data to be displayed. If None, an empty list is used.
    mask : list[np.ndarray] | None
        A list of numpy arrays representing the masks to be applied to the data. If None, an empty list is used.
    pmin : float, optional
        The minimum percentile for the color scale. Default is 1.0. If `zmin` is provided, this is ignored.
    pmax : float, optional
        The maximum percentile for the color scale. Default is 99.0. If `zmax` is provided, this is ignored.
    zmin : float | None, optional
        The minimum value for the color scale. If None, it is calculated from the data.
    zmax : float | None, optional
        The maximum value for the color scale. If None, it is calculated from the data.
    cmap : str, optional
        The color map to use for the figure. Default is 'gray'.
    binary_mode : bool, optional
        Whether to treat the data as binary strings. Default is True.
    height : int, optional
        The height of the figure in pixels. Default is 600.
    width : int, optional
        The width of the figure in pixels. Default is 600.
    align_mode : Literal['blink', 'wrap'], optional
        The alignment mode for the figure. Options are 'blink' for animation frame alignment or 'wrap' for facet column wrapping. Default is 'animate'.
    subtitles : list[str] | None
        A list of subtitles for each data and mask array. If None, default subtitles are generated.
        
    Returns
    -------
    go.Figure
        A Plotly figure object containing the data and masks visualized with the specified parameters.
    
    """
    
    # Validate inputs
    if data is None:
        data = []
    if mask is None:
        mask = []
    if len(data) + len(mask) == 0:
        raise ValueError("At least one of 'data' or 'mask' must be provided.")
    if subtitles:
        if len(subtitles) != len(data) + len(mask):
            raise ValueError("Length of 'subtitles' must match the total number of data and mask arrays.")
    else:
        subtitles = [f"Data {i+1}" for i in range(len(data))] + [f"Mask {i+1}" for i in range(len(mask))]
        
    if zmin is None:
        zmin = np.nanpercentile(np.concatenate(data), pmin)
    if zmax is None:
        zmax = np.nanpercentile(np.concatenate(data), pmax)
    
    mask_arrays = []
    for m in mask:
        mask_array = np.where(m, zmax, zmin)
        mask_arrays.append(mask_array)

    arrays_to_show = np.array(data + mask_arrays)
    match align_mode:
        case 'blink':
            align_method = {'animation_frame': 0}
        case 'wrap':
            align_method = {'facet_col': 0, 'facet_col_wrap': 2}
    fig = px.imshow(arrays_to_show,
        zmin=zmin,
        zmax=zmax,
        color_continuous_scale=cmap,
        binary_string=binary_mode,
        **align_method,
        height=height,
        width=width,
    )
    
    for annotation, subtitle in zip(fig.layout.annotations, subtitles):
        annotation.text = subtitle
    
    return fig

def plotly_sky_figure(projection_type: str = "orthographic",
                      showlatgrid: bool = True,
                      showlongrid: bool = True,
                      lataxis_dtick: int = 90,
                      lonaxis_dtick: int = 90,
                      gridcolor: str = "gray",
                      griddash: str = "dash") -> go.Figure:
    fig = go.Figure(go.Scattergeo())
    fig.update_geos(
        projection_type=projection_type,
        showland=False,
        showcoastlines=False,
        lataxis=dict(
            showgrid=showlatgrid,
            tick0=0,
            dtick=lataxis_dtick,
            gridcolor=gridcolor,
            gridwidth=2,
            griddash=griddash
        ),
        lonaxis = dict(
            showgrid=showlongrid,
            tick0=0,
            dtick=lonaxis_dtick,
            gridcolor=gridcolor,
            gridwidth=2,
            griddash=griddash
        )
    )
    fig.update_layout(margin={'b': 20, 't': 20})

    return fig
