import json
import logging
from ratelimit import limits, sleep_and_retry
from time import sleep

from quantconnect.api import *

logger = logging.getLogger(__name__)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class RateLimitedApi(Api):
    def __init__(self, user_id, token, debug = False):
        super().__init__(user_id, token, debug)

    @sleep_and_retry
    @limits(calls=20, period=60) # 20 calls per minute
    def Execute(self, endpoint, data = None, is_post = False, headers = {}):
        # TODO: Move failure logging in here?
        return super().Execute(endpoint, data, is_post, headers)

    def get_project_by_name(self, name):
        project_list = self.list_projects()
        return next((p for p in project_list["projects"] if p["name"] == name), None)

    def update_parameters_file(self, project_id, parameters):
        '''returns True or False indicating success'''
        parameters_json = json.dumps(parameters)
        # Get the file
        file_resp = self.read_project_file(project_id, "parameters.py")
        if not file_resp["success"]:
            logger.error(f"read_project_file error {file_resp}")
            return False

        # Update the content
        file_content = file_resp["files"][0]["content"]
        lines = file_content.split("\n")
        update_line = lambda line: line if line.find("_parametersJson =") < 0 else f"_parametersJson = '{parameters_json}'"
        new_lines = [update_line(line) for line in lines]
    
        # Save the new file to server
        update_resp = self.update_project_file_content(project_id, "parameters.py", "\n".join(new_lines))
        if not update_resp["success"]:
            logger.error(f"update_project_file_content error {update_resp}")
            return False

        return update_resp["success"]
    
    def compile(self, project_id, retries=5):
        '''returns compile_id or None'''
        create_compile_resp = self.create_compile(project_id)
        attempt = 0
        while attempt < retries:
            if not create_compile_resp["success"]:
                return None
            compile_id = create_compile_resp["compileId"]
            sleep(3)
            read_compile_resp = self.read_compile(project_id, compile_id)
            if not read_compile_resp["success"]:
                return None
            if read_compile_resp["state"] == "BuildSuccess":
                return compile_id
            else:
                print(f"compile state={read_compile_resp['state']}")
                attempt += 1
                create_compile_resp = self.create_compile(project_id)