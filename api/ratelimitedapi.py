import json
import logging
from ratelimit import limits, RateLimitException
import threading

from backoff import on_exception, expo
import quantconnect
from quantconnect.api import Api
import requests

logger = logging.getLogger(__name__)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

thread_local = threading.local()


def post(**kwargs):
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    new_kwargs = {**kwargs, **{"timeout": 10}}
    return thread_local.session.post(**new_kwargs)


def get(**kwargs):
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    new_kwargs = {**kwargs, **{"timeout": 10}}
    return thread_local.session.get(**new_kwargs)


setattr(quantconnect.api, "get", get)
setattr(quantconnect.api, "post", post)


class RateLimitedApi(Api):
    def __init__(self, user_id, token, debug=False):
        super().__init__(user_id, token, debug)

    @on_exception(expo, exception=(requests.exceptions.RequestException, RateLimitException), max_tries=10,
                  base=5, factor=2, max_value=10)  # kwargs that get passed to expo
    @limits(calls=20, period=5)  # 20 calls every 5 seconds
    def Execute(self, endpoint, data=None, is_post=False, headers={}):
        # TODO: Move failure logging in here?
        return super().Execute(endpoint, data, is_post, headers)

    def get_project_by_name(self, name):
        project_list = self.list_projects()
        return next((p for p in project_list["projects"] if p["name"] == name), None)

    def get_backtest_by_name(self, project_id, name):
        backtest_resp = self.list_backtests(project_id)
        backtests = backtest_resp["backtests"]

        match = [bt for bt in backtests if bt["name"] == name]
        if not match:
            return None
        if len(match) == 1:
            return match[0]
        return match

    def read_parameters_file(self, project_id):
        file_resp = self.read_project_file(project_id, "parameters.py")
        if file_resp["success"]:
            return file_resp["files"][0]["content"]
        else:
            return None

    def update_parameters_file(self, project_id, file_content, parameters, extraneous_parameters=None):
        def update_line(line):
            if line.find("_parametersJson =") >= 0:
                parameters_json = json.dumps(parameters)
                return f"_parametersJson = '{parameters_json}'"
            elif extraneous_parameters and line.find("_extraneousParametersJson =") >= 0:
                extraneous_parameters_json = json.dumps(extraneous_parameters)
                return f"_extraneousParametersJson = '{extraneous_parameters_json}'"
            else:
                return line

        lines = file_content.split("\n")
        new_lines = [update_line(line) for line in lines]

        # Save the new file to server
        update_resp = self.update_project_file_content(project_id, "parameters.py", "\n".join(new_lines))
        if not update_resp["success"]:
            logger.error(f"update_project_file_content error {update_resp}")
            return False

        return update_resp["success"]

    def read_backtest_log(self, project_id, backtest_id):
        """Read the log of a backtest in the project id specified.

        Args:
            project_id(int): Project id to read.
            backtest_id(str): Specific backtest id to read.
        Returns:
            Dictionary that contains the backtest log e.g. {"BacktestLogs": ["array", "of", "log", "lines"]}
        """
        data = {"projectId": project_id, "backtestId": backtest_id, "format": "json"}
        return self.Execute('backtests/read/log', data)
