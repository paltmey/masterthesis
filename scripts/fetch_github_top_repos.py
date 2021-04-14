import argparse
import json
import os
import requests
import sys
import time


TMP_FILE_PATH = 'tmp.json'
SEARCH_API_URL = 'https://api.github.com/search/repositories'
DELAY_AUTHENTICATED = 2
DELAY_UNAUTHENTICATED = 6


class ApiException(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message


def find_confining_name(stars, name, access_token_gen, delay):
    while True:
        total_count, items = fetch_page((stars, stars), 1, name, access_token_gen, delay)

        if total_count > 1000:
            name += 'a'
        else:
            return name, items


def get_next_name(name):
    for i in range(len(name) - 1, -1, -1):
        if name[i] == 'z':
            name = name[:i]
        else:
            name = name[:i] + chr(ord(name[i]) + 1) + name[i + 1:]
            return name

    return None


def update_repos(repos, items):
    for index, item in enumerate(items):
        stars = item['stargazers_count']
        url = item['html_url']
        repos[url] = stars


def fetch_page(star_range, page, name=None, access_token_gen=None, delay=0):
    time.sleep(delay)

    query = f'stars:{star_range[0]}..{star_range[1]}'

    if name:
        query += f' {name} in:name'

    params = {'q': query, 'sort': 'stars', 'per_page': '100', 'page': page}

    headers = {}

    access_token = next(access_token_gen)
    if access_token:
        headers['Authorization'] = f'token {access_token}'

    r = requests.get(SEARCH_API_URL, params=params, headers=headers)
    result = r.json()

    if not r.ok:
        raise ApiException(r.status_code, result['message'])

    total_count = result['total_count']
    items = result['items']

    return total_count, items


def crawl_from_last_result(star_range, repos, access_token_gen, delay):
    most_stars = star_range[0]

    while True:
        least_stars = 0

        for page in range(1, 11):
            total_count, items = fetch_page(star_range, page, access_token_gen=access_token_gen, delay=delay)

            if not items:
                break

            if page == 1:
                most_stars = items[0]['stargazers_count']
            least_stars = items[-1]['stargazers_count']

            update_repos(repos, items)

        star_range = (star_range[0], least_stars)

        print(f'Crawled {len(repos)} repositories. Lowest stars: {least_stars}')

        if most_stars == least_stars or least_stars <= star_range[0]:
            return least_stars


def crawl_decrementing_stars(star_range, repos, access_token_gen, delay):
    name = ''
    while True:
        name, items = find_confining_name(star_range[1], name, access_token_gen, delay)

        update_repos(repos, items)

        for page in range(2, 11):
            total_count, items = fetch_page((star_range[1], star_range[1]), page, name, access_token_gen, delay)

            if not items:
                break

            update_repos(repos, items)

        print(f'Crawled {len(repos)} repositories. Lowest stars: {star_range[1]}')

        name = get_next_name(name)
        if not name:
            if star_range[1] <= star_range[0]:
                return

            star_range = (star_range[0], star_range[1]-1)
            name = ''


def access_token_generator(access_tokens):
    index = 0
    while True:
        yield access_tokens[index]
        index = (index + 1) % len(access_tokens)


def crawl_repos(min_stars, max_stars, access_tokens, outfile, append):
    repos = {}

    if append and os.path.exists(outfile):
        with open(outfile) as file:
            repos = json.load(file)
            if max_stars == -1:
                max_stars = min(repos.values()) + 1  # plus one, because number of stars can vary by one from search query to pagination
            print(f'Loaded {len(repos)} repositories.')

    if max_stars == -1:
        max_stars = 500000

    access_token_gen = access_token_generator(access_tokens)
    delay = DELAY_AUTHENTICATED/len(access_tokens) if access_tokens[0] else DELAY_UNAUTHENTICATED

    try:
        print(f'Begin crawling repositories with {max_stars} to {min_stars} stars.')
        current_stars = crawl_from_last_result((min_stars, max_stars), repos, access_token_gen, delay)
        crawl_decrementing_stars((min_stars, current_stars), repos, access_token_gen, delay)

        print('Finished crawling.')
    except ApiException as err:
        print(f'Failed to fetch GitHub data with status code {err.status_code}: {err.message}.')
    except:
        print('Unexpected error:', sys.exc_info()[0])
    finally:
        with open(TMP_FILE_PATH, 'w') as file:
            print(f'Saving {len(repos)} repositories.')
            json.dump(repos, file)
            file.close()

        if os.path.exists(outfile):
            os.remove(outfile)
        os.rename(TMP_FILE_PATH, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawl most starred GitHub repositories.')
    parser.add_argument('outfile', type=str)
    parser.add_argument('--min_stars', type=int, default=50)
    parser.add_argument('--max_stars', type=int, default=-1)
    parser.add_argument('--access_tokens', type=str, nargs='*', default=[None])
    parser.add_argument('--append', action='store_true')

    args = parser.parse_args()
    crawl_repos(args.min_stars, args.max_stars, args.access_tokens, args.outfile, args.append)
