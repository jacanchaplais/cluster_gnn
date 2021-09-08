import os

import git


pwd = os.getcwd()
git_repo = git.Repo(pwd, search_parent_directories=True)
ROOT_DIR = git_repo.working_dir
