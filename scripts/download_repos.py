import argparse
import glob
import json
import os
import requests
import sys
import tarfile

from multiprocessing import Pool, Queue


class GitHubClient:
    class NotFound(Exception):
        pass

    class ApiException(Exception):
        def __init__(self, status_code, message):
            self.status_code = status_code
            self.message = message

    def __init__(self, access_tokens):
        self._access_tokens = access_tokens
        self._index = 0

    def get_default_branch(self, repo_org, repo_name):
        headers = {}
        access_token = self._get_access_token()

        if access_token:
            headers['Authorization'] = f'token {access_token}'

        repo_api_url = f'https://api.github.com/repos/{repo_org}/{repo_name}'

        response = requests.get(repo_api_url, headers=headers)
        result = response.json()

        if response.ok:
            return result['default_branch']
        elif response.status_code == 404:
            raise GitHubClient.NotFound()
        else:
            raise GitHubClient.ApiException(response.status_code, response.text)

    @staticmethod
    def get_archive(repo_org, repo_name, branch):
        archive_url = f'https://github.com/{repo_org}/{repo_name}/archive/{branch}.tar.gz'
        response = requests.get(archive_url, stream=True)

        if response.ok:
            return response.raw
        elif response.status_code == 404:
            response.close()
            raise GitHubClient.NotFound()
        else:
            raise GitHubClient.ApiException(response.status_code, response.text)

    def _get_access_token(self):
        access_token = self._access_tokens[self._index]
        self._index = (self._index + 1) % len(self._access_tokens)
        return access_token


class RepoDownloader:
    def __init__(self, output_dir, file_whitelist, github_client):
        self._output_dir = output_dir
        self._file_whitelist = tuple(file_whitelist)
        self._github_client = github_client

    def download(self, repo_url):
        repo_url_parts = repo_url.split('/')
        repo_org = repo_url_parts[3]
        repo_name = repo_url_parts[4]

        if self._repo_exists(repo_org, repo_name):
            print(f'Repository {repo_org}/{repo_name} already exists.')
        else:
            try:
                print(f'Downloading {repo_org}/{repo_name}.')
                extract_dir = os.path.join(self._output_dir, repo_org)
                branch = self._github_client.get_default_branch(repo_org, repo_name)
                stream = self._github_client.get_archive(repo_org, repo_name, branch)
                self._extract_stream(stream, extract_dir)

                output_org_dir = os.path.join(self._output_dir, repo_org)
                extracted_repo_path = self._get_extracted_repo_path(output_org_dir, repo_name)
                if extracted_repo_path:
                    flag_file_path = os.path.join(extracted_repo_path, '.complete')
                    open(flag_file_path, 'w').close()
                else:
                    print(f'Extracted folder not found for repository ##{repo_org}/{repo_name}##.', file=sys.stderr)
            except GitHubClient.NotFound:
                print(f'Repository ##{repo_org}/{repo_name}## not found.')
            except GitHubClient.ApiException as e:
                print(f'Failed fetching GitHub data: {e.status_code}: {e.message}', file=sys.stderr)
            except:
                print(f'Unexpected error while processing  ##{repo_org}/{repo_name}##: {sys.exc_info()[0]}', file=sys.stderr)

    @staticmethod
    def _included_files(members, file_whitelist):
        for tarinfo in members:
            if not file_whitelist or tarinfo.name.lower().endswith(file_whitelist):
                yield tarinfo

    def _extract_stream(self, stream, extract_dir):
        with tarfile.open(fileobj=stream, mode='r|*') as tar:
            tar.extractall(path=extract_dir, members=self._included_files(tar, self._file_whitelist))

    @staticmethod
    def _get_extracted_repo_path(output_dir, repo_name):
        repo_dir_pattern = os.path.join(output_dir, f'{repo_name}-*')
        path_list = glob.glob(repo_dir_pattern)
        if path_list and os.path.isdir(path_list[0]):
            return path_list[0]
        else:
            return None

    def _repo_exists(self, repo_org, repo_name):
        output_org_dir = os.path.join(self._output_dir, repo_org)
        extracted_repo_path = self._get_extracted_repo_path(output_org_dir, repo_name)
        if not extracted_repo_path:
            return False

        flag_file_path = os.path.join(extracted_repo_path, '.complete')
        if os.path.exists(flag_file_path):
            return True
        else:
            return False


def calc_chunksize(iterable, pool):
    print(len(pool._pool))
    chunksize, extra = divmod(len(iterable), len(pool._pool) * 4)
    if extra:
        chunksize += 1

    return chunksize


def f(repo_url):
    f.repo_downloader.download(repo_url)
    f.ready_queue.put_nowait(True)


def f_init(repo_downloader, ready_queue):
    f.ready_queue = ready_queue
    f.repo_downloader = repo_downloader


def download_and_extract_repos(infile, output_dir, file_whitelist, access_tokens, num_processes):
    with open(infile) as file:
        repos = json.load(file)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    urls = repos.keys()
    repos_size = len(urls)

    github_client = GitHubClient(access_tokens)
    repo_downloader = RepoDownloader(output_dir, file_whitelist, github_client)

    ready_queue = Queue()

    num_processed = 1
    with Pool(processes=num_processes, initializer=f_init, initargs=(repo_downloader, ready_queue)) as pool:
        result = pool.map_async(f, urls)
        while not result.ready():
            msg = ready_queue.get(timeout=1800)
            if msg:
                print(f'{num_processed}/{repos_size}')
                num_processed += 1

    ready_queue.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download most starred GitHub repositories and remove unwanted files.')
    parser.add_argument('infile', type=str)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--file_whitelist', nargs='*', default=[])
    parser.add_argument('--access_tokens', type=str, nargs='*', default=[None])
    parser.add_argument('--num_processes', type=int, default=1)

    args = parser.parse_args()
    download_and_extract_repos(args.infile, args.output_dir, args.file_whitelist, args.access_tokens, args.num_processes)
