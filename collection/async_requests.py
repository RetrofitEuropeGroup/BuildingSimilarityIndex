import asyncio
from datetime import datetime
import aiohttp
from tqdm import tqdm

### Async requests
async def fetch(session, task, headers, payload, params={}):  # fetching urls and mark result of execution
    async with session.get(task['url'], headers=headers, data=payload, params=params) as response:
        if response.status == 200:
            task['result'] =  await response.json()  # just to be sure we acquire data
            task['status'] = 'done'
        elif response.status == 429:
            retry_after = response.headers.get('retry-after')
            if retry_after is None:
                retry_after = 10
                msg = await response.text()
                print(f"Received 429, but no retry-after header. Waiting {retry_after} seconds. Message: {msg}")
            else:
                retry_after = int(retry_after)
                print(f"Received 429, waiting {retry_after} seconds for url {task['url']}")
            await asyncio.sleep(retry_after) # try again after waiting
            await fetch(session, task, headers, payload) # try again with a recursive call
        else:
            error_msg = await response.text()
            print(f"Error {response.status} while fetching url {task['url']} with params {params}. Text: {error_msg}")
            task['status'] = 'error'
            task['error message'] = error_msg

def extract_results(url_tasks):
    all_results = []
    for task in url_tasks:
        result = task.get('result')
        all_results.append(result)
        
        if result is None and task['status'] != 'error':
            print(f"ERROR: while extracting results from {task}, result is None but status is not error")
    return all_results

def tasks_to_wait(url_tasks):
    return len([i for i in url_tasks if i['status'] not in ['done', 'error']])

def tasks_done(url_tasks):
    return len([i for i in url_tasks if i['status'] in ['done', 'error']])

def tasks_active(url_tasks):
    return len([i for i in url_tasks if i['status'] == 'fetch'])

def update_bar(t, previous_progress, url_tasks):
    current_progress = tasks_done(url_tasks)
    t.update(current_progress - previous_progress)
    previous_progress = current_progress
    return previous_progress

async def request_url_list(all_urls, headers={}, payload={}, params=[], requests_persecond=50, max_active_tasks=500): #TODO: remove max_active_tasks
    try:
        async with aiohttp.ClientSession() as session:
            # convert to list of dicts
            url_tasks = [{'url': url, 'result': None, 'status': 'new'} for url in all_urls]

            started_tasks_this_second = 0
            current_second = datetime.now().second
            
            t = tqdm(total=len(url_tasks), desc='Requesting data asynchronous...')
            previous_progress = 0

            for index, task in enumerate(url_tasks):
                # always update the second, maybe we are in a new second
                if current_second != datetime.now().second:
                    while tasks_active(url_tasks) > max_active_tasks: # make sure we wait until the server is not overloaded
                        await asyncio.sleep(0.1)
                        
                    previous_progress = update_bar(t, previous_progress, url_tasks)
                    current_second = datetime.now().second
                    started_tasks_this_second = 0


                # starting the task
                url_tasks[index]['status'] = 'fetch'
                started_tasks_this_second += 1
                if len(params) > 0:
                    asyncio.create_task(fetch(session, task, headers, payload, params[index]))
                else:
                    asyncio.create_task(fetch(session, task, headers, payload))

                if started_tasks_this_second >= requests_persecond:
                    # wait until the next second so we can start new tasks
                    time_to_next_second = 1 - (datetime.now().microsecond / 1000000)
                    await asyncio.sleep(time_to_next_second)

            # loop until all tasks are done or error
            while tasks_to_wait(url_tasks) != 0:
                await asyncio.sleep(0.1)
                previous_progress = update_bar(t, previous_progress, url_tasks)
    except Exception as e:
        print(f"Error in request_url_list: {e}")

    # return results
    all_results = extract_results(url_tasks)
    return all_results
