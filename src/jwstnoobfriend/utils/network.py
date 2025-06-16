from aiohttp import ClientSession, ClientTimeout
from typing import Literal
import anyio 
import contextlib
from typing import AsyncGenerator
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
    _reference_count: int = 0
    _lock: anyio.Lock = anyio.Lock()
    _timeout: ClientTimeout | None = ClientTimeout(total=10*60)  # 10 minutes timeout for requests
    
    @classmethod
    @contextlib.asynccontextmanager
    async def session(cls) -> AsyncGenerator[ClientSession, None]:
        """
        Context manager to get an aiohttp ClientSession.
        This method ensures that the session is created only once and is reused
        across multiple calls, while also managing the reference count to close
        the session when it is no longer needed.
        
        Yields
        -------
        ClientSession
            An aiohttp ClientSession object that can be used for making requests.
        """
        async with cls._lock:
            if cls._session is None or cls._session.closed:
                cls._session = ClientSession(timeout=cls._timeout)
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
    async def fetch_json_async(
        cls,
        url: str,
        session: ClientSession,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        params: dict | None = None,
        body: dict | None = None,
    ):
        """
        Asynchronously fetch JSON data from a given URL using aiohttp.
        
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
        