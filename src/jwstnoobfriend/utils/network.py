from aiohttp import ClientSession
from typing import Literal

async def fetch_json_async(
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
        