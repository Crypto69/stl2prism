"""Talks to the Blueprint preview helper (backend.preview_helper): starts
it on the first request, feeds it one recipe at a time, restarts it when
it dies or hangs. The API server never runs the geometry kernel itself."""
import json
import logging
import os
import queue
import subprocess
import sys
import threading

log = logging.getLogger('stltosolid.preview')

_FROZEN = getattr(sys, 'frozen', False)
START_TIMEOUT_S = 90        # the first import of CadQuery on a cold NAS
BUILD_TIMEOUT_S = float(os.environ.get('STLTOSOLID_PREVIEW_TIMEOUT', 45))


class PreviewError(RuntimeError):
    readable = True

    def __init__(self, message, kind='preview', validation=None):
        super().__init__(message)
        self.kind = kind
        self.validation = validation


def _cmd():
    if _FROZEN:
        return [sys.executable, '--preview-helper']
    return [sys.executable, '-u', '-m', 'backend.preview_helper']


class Helper:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._lines = None

    def _start(self):
        self._stop()
        self._proc = subprocess.Popen(
            _cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
            cwd=None if _FROZEN else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._lines = queue.Queue()
        proc = self._proc

        def pump():
            for line in proc.stdout:
                self._lines.put(line)
            self._lines.put(None)             # EOF: the helper died

        threading.Thread(target=pump, daemon=True).start()
        first = self._read(START_TIMEOUT_S)
        if not first or not first.get('ready'):
            self._stop()
            raise PreviewError((first or {}).get('error') or
                               'The preview helper did not start (CadQuery missing?).', 'helper')

    def _stop(self):
        p, self._proc = self._proc, None
        if p is not None:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass

    def _read(self, timeout):
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            return None
        try:
            return json.loads(line)
        except ValueError:
            return {'ok': False, 'error': 'the preview helper answered nonsense'}

    def alive(self):
        return self._proc is not None and self._proc.poll() is None

    def build(self, recipe, out_dir, stem='live', timeout=BUILD_TIMEOUT_S):
        """The helper's answer dict for a valid recipe. Raises PreviewError
        with a sentence (and .validation when the recipe was refused)."""
        with self._lock:
            if not self.alive():
                self._start()
            try:
                self._proc.stdin.write(json.dumps({'recipe': recipe, 'out_dir': out_dir, 'stem': stem}) + '\n')
                self._proc.stdin.flush()
            except (OSError, ValueError):
                self._start()
                self._proc.stdin.write(json.dumps({'recipe': recipe, 'out_dir': out_dir, 'stem': stem}) + '\n')
                self._proc.stdin.flush()
            ans = self._read(timeout)
            if ans is None:
                died = not self.alive()
                self._stop()
                raise PreviewError('The preview helper crashed on this shape; it will restart for the '
                                   'next try.' if died else
                                   f'The preview took longer than {timeout:.0f} s and was stopped.', 'helper')
            if not ans.get('ok'):
                raise PreviewError(ans.get('error') or 'the preview failed', ans.get('kind', 'build'),
                                   ans.get('validation'))
            # read the mesh while the lock is still held: the next request
            # overwrites <stem>.stl, and a mesh from one build with the
            # feature map of another paints the view in stripes
            try:
                with open(os.path.join(out_dir, f'{stem}.stl'), 'rb') as f:
                    ans['stl_bytes'] = f.read()
            except OSError as e:
                raise PreviewError(f'the preview mesh could not be read ({e})', 'helper')
            return ans

    def close(self):
        with self._lock:
            self._stop()


helper = Helper()
