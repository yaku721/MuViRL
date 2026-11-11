import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


def read_records(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


@contextmanager
def record_lock(path):
    lock = path.with_name('.' + path.name + '.lock')
    started = time.monotonic()
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() - started > 60:
                raise TimeoutError(f'Record writer lock remains: {lock}')
            time.sleep(.05)
    try:
        yield
    finally:
        lock.rmdir()


def update_records(path, rows, key='id'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with record_lock(path):
        records = {row[key]: row for row in read_records(path)}
        records.update({row[key]: row for row in rows})
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records.values()))
        temporary.replace(path)
