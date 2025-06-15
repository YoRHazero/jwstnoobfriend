import inspect
import functools
from rich.progress import Progress, TaskID, track
from pydantic import BaseModel
from typing import Callable, Iterable



class InputProgressBarFunc(BaseModel):
    progress_paramkey: str | None 
    refresh_per_second: int 
    progress_description: str
def track_func(
    progress_paramkey: str | None = None, 
    refresh_per_second: int = 10, 
    progress_description: str = "Processing ...",
):
    """

    
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
    input_decorator_params = InputProgressBarFunc(**locals())
    
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


                
    
    