from jwstnoobfriend.utils.display import track_func

@track_func(progress_paramkey='items', refresh_per_second=5, progress_description='Processing items...')
def check_track_func(items: list):
    result = []
    for item in items:
        result.append(item)
    return result

if __name__ == "__main__":
    items = [i for i in range(100)]
    result = check_track_func(items)
    print(result)
    assert result == items, "The result should match the input items."