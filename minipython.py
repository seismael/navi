import os
import subprocess
env = os.environ.copy()
env.pop('VIRTUAL_ENV', None)
env.pop('UV_RUN_RECURSION_DEPTH', None)
env.pop('UV_PROJECT_ENVIRONMENT', None)
if 'PATH' in env:
    env['PATH'] = os.pathsep.join(p for p in env['PATH'].split(os.pathsep) if '.venv' not in p and 'projects\\\\auditor' not in p)
print(subprocess.run(['uv', 'run', '--project', 'projects/environment', 'python', '-c', 'import torch; print(torch.__version__)'], env=env, capture_output=True, text=True))
