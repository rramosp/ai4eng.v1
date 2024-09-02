# ------ COURSE PARAMS ------
course_id = r'ai4eng.v1\S*'
github_repo = 'rramosp/ai4eng.v1'
endpoint = 'https://m5knaekxo6.execute-api.us-west-2.amazonaws.com/dev-v0001/rlxmooc'
# ------ COURSE PARAMS ------

zip_file_url ="https://github.com/%s/archive/main.zip"%github_repo
#endpoint = 'http://localhost:5000/rlxmooc'

import subprocess
import sys
import requests, zipfile, io, os, shutil, subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def get_last_modif_date(localdir):
    try:
        import time, os, pytz
        import datetime
        k = datetime.datetime.fromtimestamp(max(os.path.getmtime(root) for root,_,_ in os.walk(localdir)))
        localtz = datetime.datetime.now(datetime.timezone(datetime.timedelta(0))).astimezone().tzinfo
        k = k.astimezone(localtz)
        return k
    except Exception:
        return None
    
def init(force_download=False):
    from IPython.display import display, HTML
    js = """
<meta name="google-signin-client_id"
      content="461673936472-kdjosv61up3ac1ajeuq6qqu72upilmls.apps.googleusercontent.com"/>
<script src="https://apis.google.com/js/client:platform.js?onload=google_button_start"></script>
    """

    display(HTML(js))

    if force_download or not os.path.exists("local"):
        print("replicating local resources")
        dirname = github_repo.split("/")[-1]+"-main/"
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        if os.path.exists("local"):
            shutil.rmtree("local")
        if os.path.exists(dirname+"/content/local"):
            shutil.move(dirname+"/content/local", "local")
        elif os.path.exists(dirname+"/local"):
            shutil.move(dirname+"/local", "local")
        shutil.rmtree(dirname)

    install("rlxutils")

def get_weblink():
    from IPython.display import HTML
    return HTML("<h3>See <a href='"+endpoint+"/web/login' target='_blank'>my courses and progress</a></h2>")


