from re import A
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from typing import Literal, Callable
import anyio 
import aiofiles
import contextlib
from typing import AsyncGenerator

from zipp import Path
class ConnectionSession:
    """
    A context manager for managing an aiohttp ClientSession.
    
    This class provides a way to create and manage an aiohttp ClientSession
    that can be used for making HTTP requests asynchronously.
    It ensures that the session is created only once and is reused across
    multiple calls, while also managing the reference count to close the
    session when it is no longer needed.
    
    Attributes
    ----------
    _session : ClientSession | None
        The aiohttp ClientSession instance.
    _reference_count : int
        The number of active references to the session.
    _lock : anyio.Lock
        A lock to ensure thread-safe access to the session.
        
    Methods
    -------
    session() -> AsyncGenerator[ClientSession, None]
        Context manager to get an aiohttp ClientSession.
    ref_count() -> int
        Get the current reference count of the session.
    fetch_json_async(
        url: str,
        session: ClientSession,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        params: dict | None = None,
        body: dict | None = None
    ) -> dict
        Asynchronously fetch JSON data from a given URL using aiohttp. 
    """
    
    _session: ClientSession | None = None
    """The aiohttp ClientSession instance."""
    _reference_count: int = 0
    """The number of active references to the session."""
    _lock: anyio.Lock = anyio.Lock()
    """A lock to ensure thread-safe access to the session."""
    _timeout: ClientTimeout | None = ClientTimeout(total=10*60)  # 10 minutes timeout for requests
    """The timeout for requests made with the session."""
    @classmethod
    @contextlib.asynccontextmanager
    async def session(cls, max_tcp_connector: int = 100) -> AsyncGenerator[ClientSession, None]:
        """
        Context manager to get an aiohttp ClientSession.
        This method ensures that the session is created only once and is reused
        across multiple calls, while also managing the reference count to close
        the session when it is no longer needed.
        
        Parameters
        ----------
        max_tcp_connector : int, optional
            The maximum number of simultaneous TCP connections allowed (default is 100). This
            set up only valid when the session is created.
        
        Yields
        -------
        ClientSession
            An aiohttp ClientSession object that can be used for making requests.
        """
        async with cls._lock:
            if cls._session is None or cls._session.closed:
                connector = TCPConnector(limit=max_tcp_connector)
                cls._session = ClientSession(timeout=cls._timeout, connector=connector)
            cls._reference_count += 1
        
        try:
            yield cls._session
        finally:
            async with cls._lock:
                cls._reference_count -= 1
                if cls._reference_count == 0:
                    if cls._session and not cls._session.closed:
                        await cls._session.close()
                    cls._session = None
    
    @classmethod
    async def ref_count(cls) -> int:
        """
        Get the current reference count of the session.
        
        Returns
        -------
        int
            The number of active references to the session.
        """
        async with cls._lock:
            return cls._reference_count
    
    @classmethod
    async def download_and_save_async(
        cls,
        url: str,
        output_path: Path,
        session: ClientSession,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        params: dict | None = None,
        body: dict | None = None,
        chunk_size: int = 1024 * 512,  # 512 KB For future development of streaming downloads and progress tracking
        progress_callback: Callable | None = None,
    ) -> AsyncGenerator[tuple[int, int], None]:
        """
        Asynchronously download content from a given URL and save it to a file.
        
        Parameters
        ----------
        url : str
            The URL to download content from.   
        output_path : Path
            The path where the downloaded content will be saved.
        session : ClientSession
            An aiohttp ClientSession object for making requests.
        method : Literal["GET", "POST", "PUT", "DELETE", "PATCH"], optional
            The HTTP method to use for the request (default is "GET").
        params : dict, optional
            Query parameters to include in the request (default is None).
        body : dict, optional
            The JSON body to send with the request (default is None).
        chunk_size : int, optional
            The size of each chunk to read from the response (default is 512 KB).
        progress_callback : Callable | None, optional
            A callback function to call when the download is complete (default is None).
            
        Yields
        -------
        AsyncGenerator[tuple[int, int], None]
            Yields tuples of (downloaded_size, total_size) if download_progress is True.
        
        Raises
        ------
        aiohttp.ClientError
            If the request fails or the response is not valid.
        """
        async with session.request(
            method,
            url,
            params=params,
            json=body,
        ) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded_size = 0
            async with aiofiles.open(output_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(chunk_size):
                    await f.write(chunk)
                    downloaded_size += len(chunk)
                    yield (downloaded_size, total_size)
                    
        if progress_callback:
            progress_callback()
    
    @classmethod
    async def fetch_content_async(
        cls,
        url: str,
        session: ClientSession,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        params: dict | None = None,
        body: dict | None = None,
    ):
        """
        Asynchronously fetch data from a given URL using aiohttp and return the response content.
        
        Parameters
        ----------
        url : str
            The URL to fetch data from.
        session : ClientSession
            An aiohttp ClientSession object for making requests.
        method : Literal["GET", "POST", "PUT", "DELETE", "PATCH"], optional
            The HTTP method to use for the request (default is "GET").
        params : dict, optional
            Query parameters to include in the request (default is None).
        body : dict, optional
            The JSON body to send with the request (default is None).
            
        Returns
        -------
        Response
            The response object from the server.
            
        Raises
        ------
        aiohttp.ClientError
            If the request fails or the response is not valid.
        """
        async with session.request(
            method,
            url,
            params=params,
            json=body,
        ) as response:
            
            response.raise_for_status()
            return await response.read()  # Return the content of the response

    @classmethod
    async def fetch_json_async(
        cls,
        url: str,
        session: ClientSession,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        params: dict | None = None,
        body: dict | None = None,
    ):
        """
        Asynchronously fetch JSON data from a given URL using aiohttp and return the parsed JSON response.
        
        Parameters
        ----------
        url : str
            The URL to fetch data from.
        session : ClientSession
            An aiohttp ClientSession object for making requests.
        method : Literal["GET", "POST", "PUT", "DELETE", "PATCH"], optional
            The HTTP method to use for the request (default is "GET").
        params : dict, optional
            Query parameters to include in the request (default is None).
        body : dict, optional
            The JSON body to send with the request (default is None).
            
        Returns
        -------
        dict
            The JSON response from the server.
            
        Raises
        ------
        aiohttp.ClientError
            If the request fails or the response is not valid JSON.
        """
        async with session.request(
            method,
            url,
            params=params,
            json=body,
        ) as response:
            
            response.raise_for_status()  # Raise an error for bad responses
            return await response.json()  # Return the JSON response
        